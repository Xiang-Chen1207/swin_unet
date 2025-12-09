"""
Script to inspect the SwinUNETR model structure and saved checkpoints.
This helps understand what parts are saved in the checkpoint.
"""

import torch
from swin_unet import SwinUNETR
from collections import OrderedDict

def print_model_structure():
    """Print the complete model structure"""
    print("=" * 80)
    print("SwinUNETR Model Structure")
    print("=" * 80)

    # Create a sample model
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=480,
        out_channels=480,
        feature_size=48,
        spatial_dims=3
    )

    print("\n### Complete Model Components ###\n")
    for name, module in model.named_children():
        print(f"├── {name}: {module.__class__.__name__}")

    print("\n### Encoder Components (Feature Extraction) ###\n")
    print("1. swinViT: Swin Transformer backbone (main encoder)")
    print("   - This is the core feature extractor")
    print("   - Extracts hierarchical features at different scales")
    print("2. encoder1-4, encoder10: Additional encoder blocks")
    print("   - Process features from swinViT at different resolutions")

    print("\n### Decoder Components (Reconstruction) ###\n")
    print("1. decoder5-1: Upsampling blocks")
    print("   - Reconstruct the original resolution")
    print("2. out: Final output layer")
    print("   - Maps to output channels (480 in your case)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for n, p in model.named_parameters() if 'swinViT' in n or 'encoder' in n)
    decoder_params = sum(p.numel() for n, p in model.named_parameters() if 'decoder' in n or 'out' in n)

    print("\n### Parameter Count ###\n")
    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"Decoder parameters: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")

    return model


def inspect_checkpoint(checkpoint_path):
    """Inspect a saved checkpoint"""
    print("\n" + "=" * 80)
    print("Checkpoint Inspection")
    print("=" * 80)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"\n### Checkpoint Keys ###\n")
    for key in checkpoint.keys():
        print(f"- {key}")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

        print(f"\n### Model State Dict Keys (showing first 20) ###\n")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            print(f"{i+1}. {key}: {state_dict[key].shape}")

        print(f"\n... (total {len(state_dict)} layers)")

        # Categorize weights
        encoder_weights = {k: v for k, v in state_dict.items() if 'swinViT' in k or 'encoder' in k}
        decoder_weights = {k: v for k, v in state_dict.items() if 'decoder' in k or 'out' in k}

        print(f"\n### Weight Categories ###\n")
        print(f"Encoder weights: {len(encoder_weights)} layers")
        print(f"Decoder weights: {len(decoder_weights)} layers")

    if 'train_loss' in checkpoint and 'val_loss' in checkpoint:
        print(f"\n### Training Info ###\n")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Train Loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
        print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")


def extract_encoder_only(checkpoint_path, output_path):
    """
    Extract only the encoder weights from a full checkpoint.
    This is useful if you want to use only the pretrained encoder for downstream tasks.
    """
    print("\n" + "=" * 80)
    print("Extracting Encoder Weights")
    print("=" * 80)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    # Extract only encoder weights (swinViT + encoder blocks)
    encoder_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if 'swinViT' in key or 'encoder' in key:
            encoder_state_dict[key] = value

    # Save encoder-only checkpoint
    encoder_checkpoint = {
        'epoch': checkpoint.get('epoch'),
        'encoder_state_dict': encoder_state_dict,
        'train_loss': checkpoint.get('train_loss'),
        'val_loss': checkpoint.get('val_loss'),
    }

    torch.save(encoder_checkpoint, output_path)

    print(f"\nEncoder weights extracted!")
    print(f"Original checkpoint: {len(state_dict)} layers")
    print(f"Encoder checkpoint: {len(encoder_state_dict)} layers")
    print(f"Saved to: {output_path}")


def show_forward_pass_info():
    """Show what happens during forward pass"""
    print("\n" + "=" * 80)
    print("Forward Pass Information")
    print("=" * 80)

    print("""
The SwinUNETR forward pass returns TWO outputs:

1. logits (reconstruction):
   - Shape: (batch, 480, 96, 96, 96)
   - This is the COMPLETE reconstruction after encoder + decoder
   - Used for computing the MAE loss

2. embedding (features):
   - Shape: (batch, 16*feature_size, 6, 6, 6) = (batch, 768, 6, 6, 6)
   - This is the deepest encoder feature (hidden_states_out[4])
   - This represents the learned representation BEFORE decoding
   - Can be used as features for downstream tasks

### What's saved in checkpoint? ###

The checkpoint saves the ENTIRE model (encoder + decoder):
- model_state_dict: Contains ALL weights
  ├── swinViT.*: Swin Transformer encoder backbone
  ├── encoder*.*: Additional encoder blocks
  ├── decoder*.*: Decoder upsampling blocks
  └── out.*: Final output layer

### Why save the full model? ###

During MAE pretraining:
1. The encoder learns to extract meaningful features (embedding)
2. The decoder learns to reconstruct from masked input
3. After pretraining, you can either:
   - Use the full model for similar reconstruction tasks
   - Extract ONLY the encoder for downstream tasks (classification, etc.)
   - Fine-tune the full model or just the encoder

### How to use encoder for downstream tasks? ###

See the extract_encoder_only() function in this script.
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect SwinUNETR model and checkpoints")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to inspect")
    parser.add_argument("--extract_encoder", action="store_true", help="Extract encoder weights only")
    parser.add_argument("--output", type=str, default="encoder_only.pth", help="Output path for encoder-only weights")

    args = parser.parse_args()

    # Always show model structure and forward pass info
    print_model_structure()
    show_forward_pass_info()

    # Inspect checkpoint if provided
    if args.checkpoint:
        inspect_checkpoint(args.checkpoint)

        if args.extract_encoder:
            extract_encoder_only(args.checkpoint, args.output)
    else:
        print("\n" + "=" * 80)
        print("Usage Examples")
        print("=" * 80)
        print("\n1. View model structure (no checkpoint needed):")
        print("   python inspect_model.py")
        print("\n2. Inspect a checkpoint:")
        print("   python inspect_model.py --checkpoint checkpoints_mae/checkpoint_epoch_1.pth")
        print("\n3. Extract encoder weights only:")
        print("   python inspect_model.py --checkpoint checkpoints_mae/checkpoint_epoch_10.pth --extract_encoder --output encoder_epoch_10.pth")
