import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms, box_convert
from matplotlib import pyplot as plt

from data import DETRData
from model import DETR
from utils.boxes import stacker
from utils.setup import get_classes
from utils.logger import get_logger
from utils.rich_handlers import TestHandler, DetectionHandler


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

TEST_DIR      = r"C:\Users\paulp\Downloads\Test"
BACKBONE_CKPT = r"D:\SignDETR\weights\resnet-weights\resnet_epoch_250.pt"
DETR_WEIGHTS  = r"D:\SignDETR\weights\detr-weights\detr_186.pt"
NUM_CLASSES   = 17
CONF_THRESH   = 0.5    # minimum confidence to consider a box
NMS_THRESH    = 0.3    # aggressive NMS — lower = fewer surviving boxes
ONE_BOX_ONLY  = True   # each image has exactly one sign — keep top-1 only
BATCH_SIZE    = 4
IMG_SIZE      = 224


# ─────────────────────────────────────────────────────────────────
# NMS FILTER
# ─────────────────────────────────────────────────────────────────

def filter_predictions(pred_logits, pred_boxes,
                        conf_thresh=0.5, nms_thresh=0.3,
                        img_size=224, one_box_only=True):
    
    probs          = pred_logits.softmax(-1)[:, :-1]  # drop no-object slot
    scores, labels = probs.max(-1)                     # (Q,), (Q,)

    # Step 1 — confidence threshold
    keep   = scores > conf_thresh
    scores = scores[keep]
    labels = labels[keep]
    boxes  = pred_boxes[keep]                          # cxcywh [0,1]

    if len(boxes) == 0:
        # Fallback — just take the single most confident prediction
        best   = probs.max(dim=0).values.argmax()       # best class overall
        scores = probs[best, probs[best].argmax()].unsqueeze(0)
        labels = probs[best].argmax().unsqueeze(0)
        boxes  = pred_boxes[best].unsqueeze(0)

    # Step 2 — convert to xyxy pixel coords
    boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    boxes_xyxy = (boxes_xyxy * img_size).clamp(0, img_size)  # clip to image bounds

    # Step 3 — cross-class NMS (boxes of ALL classes compete together)
    #           This is the key fix — per-class NMS keeps duplicates of same region
    keep_nms   = nms(boxes_xyxy.float(), scores.float(), iou_threshold=nms_thresh)
    boxes_xyxy = boxes_xyxy[keep_nms]
    labels     = labels[keep_nms]
    scores     = scores[keep_nms]

    # Step 4 — for single-object images, keep only top-1 box
    if one_box_only and len(scores) > 0:
        top1       = scores.argmax()
        boxes_xyxy = boxes_xyxy[top1].unsqueeze(0)
        labels     = labels[top1].unsqueeze(0)
        scores     = scores[top1].unsqueeze(0)

    return boxes_xyxy, labels, scores


# ─────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────

logger            = get_logger("test")
test_handler      = TestHandler()
detection_handler = DetectionHandler()
logger.print_banner()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")


CLASSES=[
    "Hello","I or Me","Okay","busy","eat","fine","forget","help",
    "how are you?","iloveyou","need","nice","no","right","same",
    "thankyou","wrong","yes"
  ]

# ─────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────

test_dataset    = DETRData(TEST_DIR, train=False)
test_dataloader = DataLoader(test_dataset, shuffle=True,
                             batch_size=BATCH_SIZE,
                             collate_fn=stacker,
                             drop_last=True)


# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────

model = DETR(
    num_classes     = NUM_CLASSES,
    backbone_ckpt   = BACKBONE_CKPT,
    freeze_backbone = False,
).to(device)

model.load_pretrained(DETR_WEIGHTS)
model.eval()
print(f"[Model] Loaded: {DETR_WEIGHTS}")


# ─────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────

X, y = next(iter(test_dataloader))
X    = X.to(device)

logger.test("Running inference...")

with torch.no_grad():
    t0             = time.time()
    result         = model(X)
    inference_time = (time.time() - t0) * 1000

print(f"[Inference] {inference_time:.1f} ms  |  batch size: {BATCH_SIZE}")
detection_handler.log_inference_time(inference_time)


# ─────────────────────────────────────────────────────────────────
# PER-IMAGE NMS FILTERING
# ─────────────────────────────────────────────────────────────────

pred_logits = result["pred_logits"].cpu()   # (B, Q, C+1)
pred_boxes  = result["pred_boxes"].cpu()    # (B, Q, 4)

all_boxes, all_labels, all_scores = [], [], []

for i in range(BATCH_SIZE):
    boxes, labels, scores = filter_predictions(
        pred_logits[i], pred_boxes[i],
        conf_thresh=CONF_THRESH,
        nms_thresh=NMS_THRESH,
        img_size=IMG_SIZE,
        one_box_only=ONE_BOX_ONLY,
    )
    all_boxes.append(boxes)
    all_labels.append(labels)
    all_scores.append(scores)
    print(f"  Image {i+1}: {len(boxes)} detection(s) after NMS")

# Log all detections
detections = []
for i in range(BATCH_SIZE):
    for box, label, score in zip(all_boxes[i], all_labels[i], all_scores[i]):
        detections.append({
            "image":      i + 1,
            "class":      CLASSES[label.item()],
            "confidence": round(score.item(), 3),
            "bbox":       [round(v, 1) for v in box.tolist()],
        })
detection_handler.log_detections(detections)


# ─────────────────────────────────────────────────────────────────
# VISUALISE
# ─────────────────────────────────────────────────────────────────

# Denormalise images for display
mean = torch.tensor([0.485, 0.456, 0.406])
std  = torch.tensor([0.229, 0.224, 0.225])
X_cpu = X.cpu()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

# Box colours per class
COLORS = [
    (0.000, 0.447, 0.741),   # blue   — class 0
    (0.850, 0.325, 0.098),   # orange — class 1
    (0.466, 0.674, 0.188),   # green  — class 2
]

for idx, (img, ax) in enumerate(zip(X_cpu, axs)):
    disp = (img.permute(1, 2, 0) * std + mean).clamp(0, 1)
    ax.imshow(disp.numpy())
    ax.set_title(f"Image {idx+1}  ({len(all_boxes[idx])} detection(s))",
                 fontsize=10)
    ax.axis("off")

    if len(all_boxes[idx]) == 0:
        ax.set_title(f"Image {idx+1}  — no detections (try lowering CONF_THRESH)",
                     fontsize=9, color="red")
        continue

    for box, label, score in zip(all_boxes[idx], all_labels[idx], all_scores[idx]):
        xmin, ymin, xmax, ymax = box.numpy()
        cls_idx = label.item()
        color   = COLORS[cls_idx % len(COLORS)]

        ax.add_patch(plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, color=color, linewidth=2))

        ax.text(xmin, max(ymin - 4, 0),
                f"{CLASSES[cls_idx]}: {score:.2f}",
                fontsize=10, color="white",
                bbox=dict(facecolor=color, alpha=0.8, pad=2))

fig.suptitle(f"Predictions  (conf>{CONF_THRESH}, NMS={NMS_THRESH}, one_box={ONE_BOX_ONLY})",
             fontsize=12)
fig.tight_layout()
plt.savefig("/kaggle/working/test_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Done] Saved → /kaggle/working/test_predictions.png")