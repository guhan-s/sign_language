import os
import sys
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DETRData
from model import DETR
from loss import DETRLoss, HungarianMatcher
from utils.logger import get_logger
from utils.boxes import stacker, box_cxcywh_to_xyxy


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

class Config:
    TRAIN_DIR      = r"C:\Users\paulp\Downloads\Train"
    TEST_DIR       = r"C:\Users\paulp\Downloads\Test"
    CHECKPOINT_DIR = r"D:\SignDETR\SignDETR-main\SignDETR-main\checkpoints"

    NUM_CLASSES    = 17
    BACKBONE_CKPT  = r"D:\SignDETR\weights\resnet-weights\resnet_epoch_250.pt"
    FREEZE_BACKBONE_EPOCHS = 10

    EPOCHS         = 120
    BATCH_SIZE     = 4
    LR             = 5e-5
    LR_BACKBONE    = 5e-6          # 10x lower for backbone (standard DETR)
    WEIGHT_DECAY   = 1e-4          # slightly more regularization than before

    GRAD_CLIP      = 0.1           # standard for DETR

    CLASS_W        = 4            # boosted: 17-class signal needs more weight
    BBOX_W         = 8
    GIOU_W         = 2
    EOS_COEF       = 0.1

    IOU_THRESHOLD  = 0.5
    SEED           = 42
    SAVE_EVERY     = 25

    RESUME_FROM    = ""  # set to "" to start fresh


# ─────────────────────────────────────────────────────────────────
# ACCURACY METRICS
# ─────────────────────────────────────────────────────────────────

def box_iou_single(box_a, box_b):
    """IoU between two boxes in xyxy format. box_a/b: (4,)"""
    x1 = torch.max(box_a[0], box_b[0])
    y1 = torch.max(box_a[1], box_b[1])
    x2 = torch.min(box_a[2], box_b[2])
    y2 = torch.min(box_a[3], box_b[3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return (inter / union.clamp(min=1e-6)).item()


def compute_detection_accuracy(yhat, y, indices, num_classes, iou_thresh=0.5):
    total_gt      = 0
    total_matched = 0
    class_correct = 0
    box_correct   = 0

    pred_logits = yhat["pred_logits"]   # (B, Q, num_classes+1)
    pred_boxes  = yhat["pred_boxes"]    # (B, Q, 4) — cxcywh, [0,1]

    for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            continue

        gt_labels = y[batch_idx]["labels"]   # (num_gt,)
        gt_boxes  = y[batch_idx]["boxes"]    # (num_gt, 4)

        total_gt      += len(gt_labels)
        total_matched += len(src_idx)

        # Predicted classes for matched queries
        matched_logits  = pred_logits[batch_idx][src_idx]   # (M, C+1)
        matched_classes = matched_logits.argmax(-1)         # (M,)
        gt_matched      = gt_labels[tgt_idx]                # (M,)
        class_correct  += (matched_classes == gt_matched).sum().item()

        # IoU for matched boxes
        matched_pred_boxes = box_cxcywh_to_xyxy(pred_boxes[batch_idx][src_idx])
        matched_gt_boxes   = box_cxcywh_to_xyxy(gt_boxes[tgt_idx])

        for pb, gb in zip(matched_pred_boxes, matched_gt_boxes):
            if box_iou_single(pb.detach().cpu(), gb.detach().cpu()) >= iou_thresh:
                box_correct += 1

    if total_matched == 0:
        return {"class_acc": 0.0, "box_acc": 0.0, "recall": 0.0}

    return {
        "class_acc": class_correct / total_matched,
        "box_acc":   box_correct   / total_matched,
        "recall":    total_matched / max(total_gt, 1),
    }


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def compute_loss(loss_dict, weight_dict):
    return (
        loss_dict["labels"]["loss_ce"]    * weight_dict["class_weighting"]
        + loss_dict["boxes"]["loss_bbox"] * weight_dict["bbox_weighting"]
        + loss_dict["boxes"]["loss_giou"] * weight_dict["giou_weighting"]
    )


def move_to_device(X, y, device):
    X = X.to(device)
    y = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
          for k, v in t.items()} for t in y]
    return X, y


# ─────────────────────────────────────────────────────────────────
# OPTIMIZER + SCHEDULER BUILDER
# Always build both together so the scheduler always tracks
# the correct optimizer — avoids the broken-scheduler bug that
# occurs when the backbone is unfrozen and a new optimizer is
# created without rebuilding the scheduler.
# ─────────────────────────────────────────────────────────────────

def build_optimizer_and_scheduler(model, cfg, frozen=True):
    if frozen:
        param_groups = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            param_groups,
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    else:
        optimizer = optim.AdamW(
            [
                {
                    "params": model.backbone.parameters(),
                    "lr": cfg.LR_BACKBONE,
                },
                {
                    "params": [p for n, p in model.named_parameters()
                               if not n.startswith("backbone")],
                    "lr": cfg.LR,
                },
            ],
            weight_decay=cfg.WEIGHT_DECAY,
        )

    # MultiStepLR: drop LR by 10x at 60% and 80% of total epochs.
    # Much more predictable than CosineAnnealingWarmRestarts on small datasets,
    # which would barely complete one cycle over 200 epochs with the old T_0.
    milestones = [int(cfg.EPOCHS * 0.6), int(cfg.EPOCHS * 0.8)]
    scheduler  = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)

    return optimizer, scheduler


# ─────────────────────────────────────────────────────────────────
# RESUME HELPER
# ─────────────────────────────────────────────────────────────────

def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load checkpoint and return (start_epoch, best_loss)."""
    if not path or not os.path.isfile(path):
        print(f"[Resume] No checkpoint at '{path}' — starting from epoch 1.")
        return 0, float("inf")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Optimizer state — only load if param group count matches
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except ValueError:
        print("[Resume] Optimizer param groups changed — skipping optimizer state.")

    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch     = ckpt.get("epoch",    0)
    best_loss = ckpt.get("val_loss", float("inf"))
    print(f"[Resume] Loaded '{path}'")
    print(f"         Resuming from epoch {epoch + 1}  |  best_loss so far: {best_loss:.5f}")
    return epoch, best_loss


# ─────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, cfg):
    model.train()
    total_loss = 0.0
    acc_sum    = {"class_acc": 0.0, "box_acc": 0.0, "recall": 0.0}

    pbar = tqdm(loader,
                desc=f"  Train [{epoch+1:>3}/{cfg.EPOCHS}]",
                unit="batch", ncols=100, leave=False)

    for batch_idx, (X, y) in enumerate(pbar):
        try:
            X, y = move_to_device(X, y, device)

            yhat = model(X)

            # Run matcher once for metrics (no extra backward cost)
            with torch.no_grad():
                indices = criterion.matcher(yhat, y)
                metrics = compute_detection_accuracy(
                    yhat, y, indices, cfg.NUM_CLASSES, cfg.IOU_THRESHOLD)

            loss_dict = criterion(yhat, y)
            loss      = compute_loss(loss_dict, criterion.weight_dict)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            for k in acc_sum:
                acc_sum[k] += metrics[k]

            n = batch_idx + 1
            pbar.set_postfix(
                loss    = f"{total_loss/n:.4f}",
                cls_acc = f"{acc_sum['class_acc']/n*100:.1f}%",
                box_acc = f"{acc_sum['box_acc']/n*100:.1f}%",
            )

        except Exception as e:
            print(f"\n[ERROR] Train epoch {epoch+1}, batch {batch_idx}: {e}")
            raise

    n = len(loader)
    return (
        total_loss / n,
        {k: v / n for k, v in acc_sum.items()},
    )


# ─────────────────────────────────────────────────────────────────
# EVAL
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, epoch, cfg):
    model.eval()
    total_loss = 0.0
    acc_sum    = {"class_acc": 0.0, "box_acc": 0.0, "recall": 0.0}

    pbar = tqdm(loader,
                desc=f"  Val   [{epoch+1:>3}/{cfg.EPOCHS}]",
                unit="batch", ncols=100, leave=False)

    for batch_idx, (X, y) in enumerate(pbar):
        X, y = move_to_device(X, y, device)
        yhat = model(X)

        indices   = criterion.matcher(yhat, y)
        metrics   = compute_detection_accuracy(
            yhat, y, indices, cfg.NUM_CLASSES, cfg.IOU_THRESHOLD)

        loss_dict = criterion(yhat, y)
        loss      = compute_loss(loss_dict, criterion.weight_dict)

        total_loss += loss.item()
        for k in acc_sum:
            acc_sum[k] += metrics[k]

        n = batch_idx + 1
        pbar.set_postfix(
            loss    = f"{total_loss/n:.4f}",
            cls_acc = f"{acc_sum['class_acc']/n*100:.1f}%",
            box_acc = f"{acc_sum['box_acc']/n*100:.1f}%",
        )

    n = len(loader)
    return (
        total_loss / n,
        {k: v / n for k, v in acc_sum.items()},
    )


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    torch.manual_seed(cfg.SEED)
    logger = get_logger("training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Data ──────────────────────────────────────────────────────
    train_dataset = DETRData(cfg.TRAIN_DIR)
    test_dataset  = DETRData(cfg.TEST_DIR, train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                              collate_fn=stacker, drop_last=True, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg.BATCH_SIZE,
                              collate_fn=stacker, drop_last=True)
    print(f"[Data] Train batches: {len(train_loader)}  |  "
          f"Test batches: {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────
    model = DETR(
        num_classes     = cfg.NUM_CLASSES,
        backbone_ckpt   = cfg.BACKBONE_CKPT,
        freeze_backbone = True,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total:,}  |  Trainable: {trainable:,}")

    # ── Loss ──────────────────────────────────────────────────────
    weight_dict = {
        "class_weighting": cfg.CLASS_W,
        "bbox_weighting":  cfg.BBOX_W,
        "giou_weighting":  cfg.GIOU_W,
    }
    matcher   = HungarianMatcher(weight_dict)
    criterion = DETRLoss(
        num_classes = cfg.NUM_CLASSES,
        matcher     = matcher,
        weight_dict = weight_dict,
        eos_coef    = cfg.EOS_COEF,
    ).to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────
    # Build with backbone frozen. Will be rebuilt at unfreeze epoch.
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, frozen=True)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────
    start_epoch, best_loss = load_checkpoint(
        cfg.RESUME_FROM, model, optimizer, scheduler, device)

    # ── Header ────────────────────────────────────────────────────
    header = (
        f"{'Ep':>4} | "
        f"{'Tr Loss':>8} {'Tr ClsAcc':>9} {'Tr BoxAcc':>9} {'Tr Recall':>9} | "
        f"{'Va Loss':>8} {'Va ClsAcc':>9} {'Va BoxAcc':>9} {'Va Recall':>9} | "
        f"{'LR':>8}"
    )
    div = "─" * len(header)
    print(f"\n{div}\n{header}\n{div}")

    # ── Loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.EPOCHS):

        # ── Backbone unfreeze ─────────────────────────────────────
        # Rebuild BOTH optimizer and scheduler together so the scheduler
        # always tracks the live optimizer. Advance the new scheduler to
        # the current epoch so milestones fire at the right time.
        if epoch == cfg.FREEZE_BACKBONE_EPOCHS:
            model.unfreeze_backbone()
            optimizer, scheduler = build_optimizer_and_scheduler(
                model, cfg, frozen=False)
            # Fast-forward scheduler to match current training progress
            for _ in range(epoch + 1):
                scheduler.step()
            print(
                f"\n[Epoch {epoch+1}] Backbone unfrozen — "
                f"backbone LR={cfg.LR_BACKBONE:.1e}, rest LR={cfg.LR:.1e}"
            )

        # ── Train ─────────────────────────────────────────────────
        train_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg)

        # ── Validate ──────────────────────────────────────────────
        val_loss, va_acc = eval_one_epoch(
            model, test_loader, criterion, device, epoch, cfg)

        # ── Scheduler step (once per epoch, after eval) ───────────
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        # ── Logging ───────────────────────────────────────────────
        is_best   = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        star      = " ★" if is_best else ""

        print(
            f"{epoch+1:>4} | "
            f"{train_loss:>8.4f} {tr_acc['class_acc']*100:>8.1f}% "
            f"{tr_acc['box_acc']*100:>8.1f}% {tr_acc['recall']*100:>8.1f}% | "
            f"{val_loss:>8.4f} {va_acc['class_acc']*100:>8.1f}% "
            f"{va_acc['box_acc']*100:>8.1f}% {va_acc['recall']*100:>8.1f}% | "
            f"{lr_now:>8.2e}"
            f"{star}"
        )

        # ── Checkpoint ────────────────────────────────────────────
        checkpoint_state = {
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss":           train_loss,
            "val_loss":             val_loss,
            "train_class_acc":      tr_acc["class_acc"],
            "val_class_acc":        va_acc["class_acc"],
            "train_box_acc":        tr_acc["box_acc"],
            "val_box_acc":          va_acc["box_acc"],
        }

        # Periodic save every SAVE_EVERY epochs → epoch_025.pt, epoch_050.pt …
        if (epoch + 1) % cfg.SAVE_EVERY == 0:
            periodic_path = os.path.join(
                cfg.CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pt")
            torch.save(checkpoint_state, periodic_path)
            print(f"       Saved → {periodic_path}")

        # Best checkpoint (val loss improved)
        if is_best:
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "best.pt")
            torch.save(checkpoint_state, best_path)
            print(f"       ★ Best saved → {best_path}  "
                  f"(val_loss={val_loss:.5f})")

        # Latest checkpoint (always overwrite)
        final_path = os.path.join(cfg.CHECKPOINT_DIR, "final.pt")
        torch.save(checkpoint_state, final_path)

    print(div)
    print(f"\n[Done] Best val loss : {best_loss:.5f}")
    print(f"[Done] Checkpoints   : {cfg.CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()