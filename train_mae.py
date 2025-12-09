
import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from swin_unet import SwinUNETR
from tqdm import tqdm

class FMRIDataset(Dataset):
    def __init__(self, data_list_file, time_window=480, transform=None):
        """
        Args:
            data_list_file: Path to text file containing subject folder paths (one per line).
            time_window: Number of timepoints to use as channels.
            transform: Optional transform to be applied on a sample.
        """
        self.time_window = time_window
        self.transform = transform

        # Read subject paths from text file
        self.subject_paths = []
        with open(data_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    self.subject_paths.append(line)

        print(f"Loaded {len(self.subject_paths)} subjects from {data_list_file}")

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, idx):
        sub_dir = self.subject_paths[idx]
        
        # Find all .npz files in the subject directory
        npz_files = [f for f in os.listdir(sub_dir) if f.endswith('.npz')]
        npz_files.sort() # Sort to ensure correct temporal order
        
        data_list = []
        
        # Load all found npz files
        for file_name in npz_files:
            path = os.path.join(sub_dir, file_name)
            try:
                # Use mmap_mode='r' to avoid loading the entire file into memory
                data_file = np.load(path, mmap_mode='r')
                # If it's a NpzFile object, we need to access the array
                if isinstance(data_file, np.lib.npyio.NpzFile):
                    # Assume the first key holds the data
                    key = data_file.files[0]
                    data = data_file[key]
                else:
                    data = data_file
                
                # Ensure data is loaded into memory for concatenation
                data = np.array(data)
                data_list.append(data)
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Continue to next file or handle error
                continue

        if not data_list:
            print(f"No data found for {sub_dir}")
            return torch.zeros((self.time_window, 96, 96, 96))

        # Concatenate along the last dimension (time)
        # Each data is expected to be (96, 96, 96, T_part)
        try:
            sample = np.concatenate(data_list, axis=-1) # (96, 96, 96, T_total)
        except ValueError as e:
            print(f"Error concatenating data for {sub_dir}: {e}")
            return torch.zeros((self.time_window, 96, 96, 96))
        
        total_timepoints = sample.shape[-1]
        
        if total_timepoints < self.time_window:
            # Pad if not enough timepoints (should not happen if 12 * 40 = 480)
            # For simplicity, just return what we have or pad with zeros
            # But here we expect exactly 480 usually.
            pass
        else:
            # If we have more than needed (unlikely here), crop. 
            # If exact, this just takes the whole thing.
            start_t = 0
            if total_timepoints > self.time_window:
                start_t = np.random.randint(0, total_timepoints - self.time_window + 1)
            sample = sample[..., start_t : start_t + self.time_window]
        
        # sample shape: (96, 96, 96, time_window)
        # Transpose to (time_window, 96, 96, 96) -> (C, D, H, W)
        sample = sample.transpose(3, 0, 1, 2)
        
        # Convert to float32 tensor
        sample = torch.from_numpy(sample).float()
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class MaskGenerator:
    def __init__(self, input_size, mask_ratio=0.75, patch_size=(4, 4, 4)):
        """
        Args:
            input_size: (D, H, W)
            mask_ratio: Ratio of patches to mask.
            patch_size: Size of the patches to mask.
        """
        self.input_size = input_size
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        self.D, self.H, self.W = input_size
        self.pD, self.pH, self.pW = patch_size
        
        assert self.D % self.pD == 0 and self.H % self.pH == 0 and self.W % self.pW == 0, \
            "Input size must be divisible by patch size."
            
        self.num_patches_d = self.D // self.pD
        self.num_patches_h = self.H // self.pH
        self.num_patches_w = self.W // self.pW
        self.num_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w

    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            masked_x: Tensor with masked regions set to 0.
            mask: Binary mask tensor of shape (B, 1, D, H, W), 1 where masked.
        """
        B, C, D, H, W = x.shape
        
        mask = np.zeros((B, self.num_patches), dtype=int)
        num_masked = int(self.num_patches * self.mask_ratio)
        mask[:, :num_masked] = 1
        
        # Shuffle each row independently
        for i in range(B):
            np.random.shuffle(mask[i])
        
        mask = mask.reshape((B, self.num_patches_d, self.num_patches_h, self.num_patches_w))
        
        # Upsample mask to original size
        # Repeat elements
        mask = mask.repeat(self.pD, axis=1).repeat(self.pH, axis=2).repeat(self.pW, axis=3)
        
        mask = torch.from_numpy(mask).to(x.device).unsqueeze(1) # (B, 1, D, H, W)
        
        masked_x = x * (1 - mask)
        
        return masked_x, mask

def train(args):
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Training Dataset and DataLoader
    train_dataset = FMRIDataset(
        data_list_file=args.train_list,
        time_window=args.time_window
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Validation Dataset and DataLoader
    val_dataset = FMRIDataset(
        data_list_file=args.val_list,
        time_window=args.time_window
    )

    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    # in_channels = time_window
    # out_channels = time_window (reconstruction)
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=args.time_window,
        out_channels=args.time_window,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        spatial_dims=3
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Mask Generator
    mask_generator = MaskGenerator(
        input_size=(96, 96, 96),
        mask_ratio=args.mask_ratio,
        patch_size=tuple(args.mask_patch_size)
    )
    
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
        print(f"Starting training on {device}...")

        # Initialize CSV log file
        csv_log_path = os.path.join(args.output_dir, 'loss_log.csv')
        with open(csv_log_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['epoch', 'train_loss', 'val_loss'])
    else:
        writer = None
    
    criterion = nn.MSELoss(reduction='none')

    for epoch in range(args.epochs):
        # ============ Training Phase ============
        train_sampler.set_epoch(epoch)
        model.train()
        train_epoch_loss = 0

        if dist.get_rank() == 0:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        else:
            progress_bar = train_dataloader

        embedding = None
        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)

            # Generate mask
            masked_batch, mask = mask_generator(batch)

            # Forward pass
            outputs, embedding = model(masked_batch)

            # Compute loss
            # Loss is computed only on masked regions (SimMIM style) or all (standard AE)
            # MAE usually computes loss only on masked patches.

            loss = criterion(outputs, batch)

            if args.loss_on_masked_only:
                loss = (loss * mask).sum() / (mask.sum() + 1e-6)
            else:
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Reduce loss for logging
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()

            train_epoch_loss += loss.item()
            if dist.get_rank() == 0:
                progress_bar.set_postfix({"loss": loss.item()})

                # Log loss to TensorBoard
                global_step = epoch * len(train_dataloader) + step
                writer.add_scalar('Loss/train', loss.item(), global_step)

        avg_train_loss = train_epoch_loss / len(train_dataloader)

        # ============ Validation Phase ============
        model.eval()
        val_epoch_loss = 0

        if dist.get_rank() == 0:
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        else:
            val_progress_bar = val_dataloader

        with torch.no_grad():
            for step, batch in enumerate(val_progress_bar):
                batch = batch.to(device)

                # Generate mask
                masked_batch, mask = mask_generator(batch)

                # Forward pass
                outputs, _ = model(masked_batch)

                # Compute loss
                loss = criterion(outputs, batch)

                if args.loss_on_masked_only:
                    loss = (loss * mask).sum() / (mask.sum() + 1e-6)
                else:
                    loss = loss.mean()

                # Reduce loss for logging
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()

                val_epoch_loss += loss.item()
                if dist.get_rank() == 0:
                    val_progress_bar.set_postfix({"val_loss": loss.item()})

        avg_val_loss = val_epoch_loss / len(val_dataloader)

        # ============ Logging and Checkpointing ============
        if dist.get_rank() == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # Log to TensorBoard
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)

            # Write losses to CSV
            with open(csv_log_path, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

            # Save checkpoint every epoch
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinUNETR MAE Pretraining for fMRI")

    parser.add_argument("--train_list", type=str, default="/public/home/wangmo/swinunet/pretrain/train.txt", help="Path to text file containing training subject paths")
    parser.add_argument("--val_list", type=str, default="/public/home/wangmo/swinunet/pretrain/val.txt", help="Path to text file containing validation subject paths")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--time_window", type=int, default=480, help="Number of timepoints to use as channels (12 files * 40 = 480)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--feature_size", type=int, default=48, help="Feature size for SwinUNETR")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Masking ratio")
    parser.add_argument("--mask_patch_size", type=int, nargs=3, default=[4, 4, 4], help="Patch size for masking")
    parser.add_argument("--loss_on_masked_only", action="store_true", default=True, help="Compute loss only on masked regions")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
