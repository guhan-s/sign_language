import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class Config:
    DATA_DIR      = "/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200"
    OUTPUT_DIR    = "/kaggle/working/checkpoints"
    RESUME_FROM   = "/kaggle/working/checkpoints/epoch_240.pth"   

    NUM_CLASSES   = 200
    EPOCHS        = 250
    BATCH_SIZE    = 128
    LR            = 0.1          # SGD works better with 0.1 than 0.01 for ResNets
    MOMENTUM      = 0.9
    WEIGHT_DECAY  = 1e-4
    LR_MILESTONES = [100, 150, 200]   # 3-step decay for 250 epochs
    LR_GAMMA      = 0.1
    SAVE_EVERY    = 10

    NUM_WORKERS   = 2
    IMG_SIZE      = 64
    SEED          = 42


# ─────────────────────────────────────────────────────────────────
# BOTTLENECK BLOCK  (the key difference vs your BasicBlock)
# ─────────────────────────────────────────────────────────────────
# ResNet50 uses Bottleneck (1x1 → 3x3 → 1x1) instead of BasicBlock
# (3x3 → 3x3). This gives:
#   - More non-linearity per parameter
#   - Better gradient flow through the 1x1 shortcuts
#   - Richer feature representations at the same compute cost

class Bottleneck(nn.Module):
    expansion = 4   # output channels = base_channels * 4

    def __init__(self, in_channels, base_channels, stride=1):
        super().__init__()
        # 1x1 — reduce channels
        self.conv1 = nn.Conv2d(in_channels, base_channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(base_channels)
        # 3x3 — spatial reasoning
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(base_channels)
        # 1x1 — expand channels
        self.conv3 = nn.Conv2d(base_channels, base_channels * self.expansion,
                               1, bias=False)
        self.bn3   = nn.BatchNorm2d(base_channels * self.expansion)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != base_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, base_channels * self.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(base_channels * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ─────────────────────────────────────────────────────────────────
# RESNET50-STYLE MODEL  (adapted for 64x64 TinyImageNet input)
# ─────────────────────────────────────────────────────────────────

class ResNet50Style(nn.Module):
    """
    ResNet50-equivalent depth and feature quality, adapted for
    64×64 TinyImageNet input.

    Stage layout matches ResNet50: [3, 4, 6, 3] bottleneck blocks.
    Output feature map is 2048-d, same as ResNet50 — so it drops
    directly into your existing DETR backbone slot with no changes.

    Key differences from your ModifiedResNet18:
      - Bottleneck blocks (3 convs + residual) instead of BasicBlock
      - [3,4,6,3] blocks per stage instead of [2,2,2,2]
      - No fake expand / feature_enhance layers
      - Stem adapted for 64×64 (no 7×7 stride-2 + maxpool)
      - Dropout before classifier to reduce overfitting on small data
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.in_channels = 64

        # ── Stem (adapted for 64×64, no aggressive downsampling) ──
        # Real ResNet50 uses 7×7 stride-2 + maxpool for 224×224.
        # For 64×64 we use a 3-conv stem with stride-1 to preserve
        # spatial resolution in early layers.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Residual stages — [3, 4, 6, 3] matches ResNet50 ───────
        # Output channels after each stage: 256, 512, 1024, 2048
        self.layer1 = self._make_layer(base_channels=64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(base_channels=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(base_channels=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(base_channels=512, blocks=3, stride=2)
        # Final feature map: (B, 2048, 8, 8) for 64×64 input

        # ── Head ────────────────────────────────────────────────────
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(p=0.3)          # helps on small datasets
        self.classifier = nn.Linear(2048, num_classes)

        self._init_weights()

    def _make_layer(self, base_channels, blocks, stride):
        layers = [Bottleneck(self.in_channels, base_channels, stride)]
        self.in_channels = base_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, base_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)             # (B,   64, 64, 64)
        x = self.layer1(x)           # (B,  256, 64, 64)
        x = self.layer2(x)           # (B,  512, 32, 32)
        x = self.layer3(x)           # (B, 1024, 16, 16)
        x = self.layer4(x)           # (B, 2048,  8,  8)
        x = self.avgpool(x)          # (B, 2048,  1,  1)
        x = torch.flatten(x, 1)      # (B, 2048)
        x = self.dropout(x)
        x = self.classifier(x)       # (B, num_classes)
        return x

    def get_feature_maps(self, x):
        """
        Returns intermediate feature maps for use as DETR backbone.
        Call this instead of forward() in your DETR model.
        Returns the layer4 output: (B, 2048, H/8, W/8)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x   # (B, 2048, H/8, W/8) — no pooling, spatial info preserved


# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────

class TinyImageNetDataset(Dataset):

    def __init__(self, root: str, split: str = "train", transform=None):
        assert split in ("train", "val")
        self.root      = Path(root)
        self.split     = split
        self.transform = transform

        with open(self.root / "wnids.txt") as f:
            wnids = sorted(line.strip() for line in f if line.strip())
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}
        self.samples      = self._load_samples()

    def _load_samples(self):
        samples = []
        if self.split == "train":
            for wnid, idx in tqdm(list(self.class_to_idx.items()),
                                  desc="  Scanning train", unit="class", ncols=80):
                img_dir = self.root / "train" / wnid / "images"
                for p in img_dir.glob("*.JPEG"):
                    samples.append((str(p), idx))
        else:
            annot = self.root / "val" / "val_annotations.txt"
            with open(annot) as f:
                lines = f.readlines()
            for line in tqdm(lines, desc="  Scanning val  ",
                             unit="img", ncols=80):
                fname, wnid, *_ = line.strip().split("\t")
                if wnid in self.class_to_idx:
                    p = self.root / "val" / "images" / fname
                    samples.append((str(p), self.class_to_idx[wnid]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(cfg: Config):
    mean = [0.4802, 0.4481, 0.3975]
    std  = [0.2770, 0.2691, 0.2821]

    train_tf = transforms.Compose([
        transforms.RandomCrop(cfg.IMG_SIZE, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        # RandAugment gives a meaningful boost with no tuning needed
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    print("[Data] Scanning train split...")
    train_ds = TinyImageNetDataset(cfg.DATA_DIR, "train", train_tf)
    print("[Data] Scanning val split...")
    val_ds   = TinyImageNetDataset(cfg.DATA_DIR, "val",   val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE * 2,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)

    print(f"[Data] Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ─────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str, is_best: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_best:
        best = os.path.join(os.path.dirname(path), "best.pth")
        shutil.copyfile(path, best)


def load_checkpoint(path, model, optimizer, scheduler, device):
    if not path or not os.path.isfile(path):
        print("[Ckpt] Starting from scratch.\n")
        return 0, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch    = ckpt.get("epoch",    0)
    best_acc = ckpt.get("best_acc", 0.0)
    print(f"[Ckpt] Resumed <- {path}  (epoch {epoch}, best {best_acc:.2f}%)\n")
    return epoch, best_acc


# ─────────────────────────────────────────────────────────────────
# TRAIN / VALIDATE
# ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, cfg):
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(loader,
                desc=f"  Train [{epoch+1:>3}/{cfg.EPOCHS}]",
                unit="batch", ncols=95, leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += logits.detach().argmax(1).eq(labels).sum().item()
        total      += images.size(0)

        pbar.set_postfix(
            loss=f"{total_loss / total:.4f}",
            acc =f"{100. * correct / total:.2f}%",
        )

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, cfg):
    model.eval()
    total_loss = top1 = top5 = total = 0

    pbar = tqdm(loader,
                desc=f"  Val   [{epoch+1:>3}/{cfg.EPOCHS}]",
                unit="batch", ncols=95, leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits      = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        total      += images.size(0)
        top1       += logits.argmax(1).eq(labels).sum().item()
        _, pred5    = logits.topk(5, dim=1)
        top5       += pred5.eq(labels.unsqueeze(1)).any(1).sum().item()

        pbar.set_postfix(
            top1=f"{100. * top1 / total:.2f}%",
            top5=f"{100. * top5 / total:.2f}%",
        )

    return total_loss / total, 100. * top1 / total, 100. * top5 / total


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    torch.manual_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_loader, val_loader = get_dataloaders(cfg)

    model  = ResNet50Style(num_classes=cfg.NUM_CLASSES).to(device)
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] ResNet50Style | total: {total:,} | trainable: {trainable:,}")

    # Label smoothing helps on TinyImageNet (200 fine-grained classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.LR,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.LR_MILESTONES, gamma=cfg.LR_GAMMA)

    start_epoch, best_acc = load_checkpoint(
        cfg.RESUME_FROM, model, optimizer, scheduler, device)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    header  = (f"{'Epoch':>7} | "
               f"{'Train Loss':>10} {'Train Acc':>10} | "
               f"{'Val Loss':>9} {'Val Top-1':>9} {'Val Top-5':>9} | "
               f"{'LR':>8} {'Time':>7}")
    divider = "─" * len(header)
    print(f"\n{divider}\n{header}\n{divider}")

    for epoch in range(start_epoch, cfg.EPOCHS):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg)

        val_loss, val_top1, val_top5 = validate(
            model, val_loader, criterion, device, epoch, cfg)

        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        is_best  = val_top1 > best_acc
        best_acc = max(best_acc, val_top1)
        star     = "  ★" if is_best else ""

        print(
            f"{epoch+1:>7} | "
            f"{tr_loss:>10.4f} {tr_acc:>9.2f}% | "
            f"{val_loss:>9.4f} {val_top1:>8.2f}% {val_top5:>8.2f}% | "
            f"{lr_now:>8.1e} {elapsed:>6.1f}s"
            f"{star}"
        )

        if (epoch + 1) % cfg.SAVE_EVERY == 0 or is_best \
                or (epoch + 1) == cfg.EPOCHS:
            ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch+1:03d}.pth")
            save_checkpoint(
                state={
                    "epoch":                epoch + 1,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc":             best_acc,
                    "val_top1":             val_top1,
                    "val_top5":             val_top5,
                },
                path=ckpt_path,
                is_best=is_best,
            )
            tag = " (best.pth updated)" if is_best else ""
            print(f"        Saved: {ckpt_path}{tag}")

    print(divider)
    print(f"\n[Done] Best Val Top-1 : {best_acc:.2f}%")
    print(f"[Done] Checkpoints    : {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()