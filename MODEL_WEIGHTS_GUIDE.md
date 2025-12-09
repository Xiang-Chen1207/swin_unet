# SwinUNETR 模型权重说明文档

## 目录
1. [模型结构概述](#模型结构概述)
2. [保存的权重包含什么](#保存的权重包含什么)
3. [Forward输出说明](#forward输出说明)
4. [如何使用保存的权重](#如何使用保存的权重)
5. [如何提取Encoder权重](#如何提取encoder权重)

---

## 模型结构概述

SwinUNETR是一个**编码器-解码器（Encoder-Decoder）架构**，用于MAE预训练：

```
输入 (480, 96, 96, 96)
    ↓
┌─────────────────────────────────┐
│  ENCODER (特征提取)              │
│  ├── swinViT                    │  ← Swin Transformer骨干网络
│  │   ├── patch_embed            │  ← 将输入分块并嵌入
│  │   ├── layers1-4               │  ← 4个Swin层，逐层下采样
│  │   └── 输出多尺度特征          │
│  ├── encoder1                    │  ← 额外的编码块（分辨率：96³）
│  ├── encoder2                    │  ← 额外的编码块（分辨率：48³）
│  ├── encoder3                    │  ← 额外的编码块（分辨率：24³）
│  ├── encoder4                    │  ← 额外的编码块（分辨率：12³）
│  └── encoder10                   │  ← 最深层编码块（分辨率：6³）
└─────────────────────────────────┘
    ↓
  Embedding (768, 6, 6, 6)  ← 这是最深层的特征表示
    ↓
┌─────────────────────────────────┐
│  DECODER (重建)                  │
│  ├── decoder5                    │  ← 上采样（6³ → 12³）
│  ├── decoder4                    │  ← 上采样（12³ → 24³）
│  ├── decoder3                    │  ← 上采样（24³ → 48³）
│  ├── decoder2                    │  ← 上采样（48³ → 96³）
│  ├── decoder1                    │  ← 最终上采样
│  └── out                         │  ← 输出层
└─────────────────────────────────┘
    ↓
输出 logits (480, 96, 96, 96)  ← 重建的完整数据
```

### 参数统计

对于你的配置（feature_size=48, in_channels=480）：

- **总参数**: ~62M
  - **Encoder参数**: ~38M (61%)
  - **Decoder参数**: ~24M (39%)

---

## 保存的权重包含什么

在 `train_mae.py:322-329`，每个epoch保存的checkpoint包含：

```python
checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': model.module.state_dict(),  # ← 完整模型权重
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'val_loss': avg_val_loss,
}
```

### model_state_dict 包含的所有权重：

```
model_state_dict:
├── swinViT.patch_embed.proj.weight         # Encoder: Patch embedding
├── swinViT.patch_embed.proj.bias
├── swinViT.layers1.0.blocks.0.*            # Encoder: Swin layer 1
├── swinViT.layers2.0.blocks.0.*            # Encoder: Swin layer 2
├── swinViT.layers3.0.blocks.0.*            # Encoder: Swin layer 3
├── swinViT.layers4.0.blocks.0.*            # Encoder: Swin layer 4
├── encoder1.layer.conv1.conv.weight        # Encoder: 额外编码块
├── encoder2.layer.conv1.conv.weight
├── encoder3.layer.conv1.conv.weight
├── encoder4.layer.conv1.conv.weight
├── encoder10.layer.conv1.conv.weight       # Encoder: 最深层
├── decoder5.transp_conv.conv.weight        # Decoder: 上采样块
├── decoder4.transp_conv.conv.weight
├── decoder3.transp_conv.conv.weight
├── decoder2.transp_conv.conv.weight
├── decoder1.transp_conv.conv.weight
└── out.conv.conv.weight                    # Decoder: 输出层
```

**结论**: checkpoint保存了**完整的模型**（Encoder + Decoder）

---

## Forward输出说明

查看 `swin_unet.py:303-316`，forward函数返回两个值：

```python
def forward(self, x_in):
    hidden_states_out = self.swinViT(x_in, self.normalize)  # Encoder提取特征

    # Encoder处理
    enc0 = self.encoder1(x_in)
    enc1 = self.encoder2(hidden_states_out[0])
    enc2 = self.encoder3(hidden_states_out[1])
    enc3 = self.encoder4(hidden_states_out[2])
    dec4 = self.encoder10(hidden_states_out[4])  # 最深层encoder特征

    # Decoder重建
    dec3 = self.decoder5(dec4, hidden_states_out[3])
    dec2 = self.decoder4(dec3, enc3)
    dec1 = self.decoder3(dec2, enc2)
    dec0 = self.decoder2(dec1, enc1)
    out = self.decoder1(dec0, enc0)
    logits = self.out(out)

    return logits, hidden_states_out[4]
    #      ^^^^^^  ^^^^^^^^^^^^^^^^^^^
    #      重建输出  Encoder最深层特征（embedding）
```

### 返回值说明：

1. **logits**: `(batch, 480, 96, 96, 96)`
   - 完整的重建输出
   - 经过了 Encoder + Decoder
   - 用于计算MAE损失（与原始输入比较）
   - 这是**掩码重建任务的最终输出**

2. **embedding**: `(batch, 768, 6, 6, 6)`
   - Encoder最深层的特征表示
   - 就是 `hidden_states_out[4]`（encoder10的输入）
   - 维度：768 = 16 × feature_size = 16 × 48
   - 这是**压缩的特征表示**，包含了输入的语义信息
   - 可用于下游任务（分类、回归等）

---

## 如何使用保存的权重

### 场景1: 继续MAE预训练

```python
# 加载完整模型继续训练
checkpoint = torch.load('checkpoint_epoch_10.pth')

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=480,
    out_channels=480,
    feature_size=48,
    spatial_dims=3
)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']

# 继续训练...
```

### 场景2: 使用完整模型做重建任务

```python
# 加载模型用于推理
checkpoint = torch.load('checkpoint_epoch_10.pth')

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=480,
    out_channels=480,
    feature_size=48,
    spatial_dims=3
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    reconstructed, embedding = model(input_data)
    # reconstructed: 重建结果
    # embedding: 特征表示
```

### 场景3: 只使用Encoder做下游任务（推荐）

这是最常见的场景！预训练的**Encoder**学到了有用的特征表示，可以用于：
- 分类任务
- 回归任务
- 其他下游任务

```python
# 1. 加载完整checkpoint
checkpoint = torch.load('checkpoint_epoch_10.pth')
full_state_dict = checkpoint['model_state_dict']

# 2. 提取encoder权重
encoder_state_dict = {k: v for k, v in full_state_dict.items()
                      if 'swinViT' in k or 'encoder' in k}

# 3. 创建只包含encoder的模型
model = SwinUNETR(...)
model.load_state_dict(encoder_state_dict, strict=False)  # strict=False允许部分加载

# 4. 冻结encoder，只训练新的分类头
for name, param in model.named_parameters():
    if 'swinViT' in name or 'encoder' in name:
        param.requires_grad = False

# 5. 添加新的分类头
classifier = nn.Linear(768 * 6 * 6 * 6, num_classes)  # 768是embedding的channel数

# 6. Fine-tune...
```

---

## 如何提取Encoder权重

我已经为你创建了工具脚本：`inspect_model.py`

### 使用方法：

#### 1. 查看模型结构
```bash
python inspect_model.py
```

#### 2. 检查checkpoint内容
```bash
python inspect_model.py --checkpoint checkpoints_mae/checkpoint_epoch_1.pth
```

#### 3. 提取encoder权重
```bash
python inspect_model.py \
    --checkpoint checkpoints_mae/checkpoint_epoch_10.pth \
    --extract_encoder \
    --output encoder_only_epoch_10.pth
```

这会生成一个只包含encoder权重的文件，文件大小会小约40%。

---

## 总结

| 问题 | 答案 |
|------|------|
| checkpoint保存了什么？ | **完整模型**（Encoder + Decoder） |
| forward返回的logits是什么？ | **重建输出**（经过Encoder+Decoder） |
| forward返回的embedding是什么？ | **Encoder最深层特征** (768, 6, 6, 6) |
| 哪部分是"Encoder权重"？ | 所有包含`swinViT`或`encoder`的层 |
| 下游任务应该用什么？ | 通常只用**Encoder部分** + 新的任务头 |
| 为什么保存完整模型？ | 灵活性：可以继续MAE训练，也可以提取encoder |

---

## 推荐工作流程

```
MAE预训练（当前）
    ↓
保存完整checkpoint (Encoder + Decoder)
    ↓
    ├─→ 继续预训练：使用完整checkpoint
    ├─→ 重建任务：使用完整模型
    └─→ 下游任务：提取Encoder + 添加任务头
```

MAE预训练的**核心价值**在于Encoder学到的特征表示，Decoder只是用来辅助训练的。

---

如有疑问，请运行：
```bash
python inspect_model.py --help
```
