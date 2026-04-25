import math
import os
import torch
import torch.nn as nn
from torchinfo import summary
from resnet_scratch import ResNet50Style          


# ─────────────────────────────────────────────────────────────────
# Positional encoding (unchanged)
# ─────────────────────────────────────────────────────────────────

def _get_1d_sincos_pos_embed(length: int, dim: int,
                              temperature: float = 10000.0, device=None):
    assert dim % 2 == 0
    position = torch.arange(length, device=device,
                             dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(temperature) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def build_2d_sincos_position_embedding(height: int, width: int,
                                        dim: int, device=None):
    assert dim % 2 == 0
    dim_half = dim // 2
    pe_y = _get_1d_sincos_pos_embed(height, dim_half, device=device)
    pe_x = _get_1d_sincos_pos_embed(width,  dim_half, device=device)
    pos  = torch.zeros(height, width, dim, device=device, dtype=torch.float32)
    pos[:, :, :dim_half] = pe_y[:, None, :].expand(-1, width, -1)
    pos[:, :, dim_half:] = pe_x[None, :, :].expand(height, -1, -1)
    return pos.view(1, height * width, dim)


# ─────────────────────────────────────────────────────────────────
# Backbone wrapper
# ─────────────────────────────────────────────────────────────────

class ResNet50FeatureExtractor(nn.Module):
    """
    Wraps ResNet50Style and returns spatial feature maps (B, 2048, H/8, W/8)
    by calling the existing get_feature_maps() method — skips avgpool,
    dropout and classifier which are not needed for detection.
    """

    def __init__(self, backbone: ResNet50Style):
        super().__init__()
        # Copy every stage directly — no head layers included
        self.stem   = backbone.stem      # (B,   64, H,   W  )
        self.layer1 = backbone.layer1    # (B,  256, H,   W  )  ← Bottleneck ×3
        self.layer2 = backbone.layer2    # (B,  512, H/2, W/2)  ← Bottleneck ×4
        self.layer3 = backbone.layer3    # (B, 1024, H/4, W/4)  ← Bottleneck ×6
        self.layer4 = backbone.layer4    # (B, 2048, H/8, W/8)  ← Bottleneck ×3
        # avgpool / dropout / classifier are intentionally NOT copied

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x    # (B, 2048, H/8, W/8) — spatial map ready for DETR


# ─────────────────────────────────────────────────────────────────
# DETR
# ─────────────────────────────────────────────────────────────────

class DETR(nn.Module):
    def __init__(
        self,
        num_classes:         int  = 17,
        backbone_ckpt:       str  = r"D:\SignDETR\weights\resnet-weights\resnet_epoch_250.pt",
        hidden_dim:          int  = 256,
        nheads:              int  = 8,
        num_encoder_layers:  int  = 1,
        num_decoder_layers:  int  = 1,
        num_queries:         int  = 25,
        freeze_backbone:     bool = True,
    ):
        super().__init__()

        # ── 1. Build ResNet50Style and load scratch-trained weights ──
        _full_model = ResNet50Style(num_classes=200)   # same num_classes as pretraining

        if backbone_ckpt and os.path.isfile(backbone_ckpt):
            raw    = torch.load(backbone_ckpt, map_location="cpu")
            ckpt_sd = raw["model_state_dict"] if "model_state_dict" in raw else raw

            # Drop classifier & avgpool keys — DETR never uses them
            backbone_sd = {
                k: v for k, v in ckpt_sd.items()
                if not k.startswith("classifier")
                and not k.startswith("avgpool")
                and not k.startswith("dropout")
            }
            missing, unexpected = _full_model.load_state_dict(
                backbone_sd, strict=False)
            print(f"[Backbone] Loaded pretrained weights from: {backbone_ckpt}")
            print(f"           Missing   (expected — head layers) : {missing}")
            print(f"           Unexpected (should be empty)       : {unexpected}")
        else:
            print("[Backbone] No checkpoint found — backbone is randomly initialised.")

        # ── 2. Wrap backbone to return spatial feature maps ──────────
        # Uses ResNet50FeatureExtractor which copies the 5 stages only.
        # ResNet50Style.get_feature_maps() could also be used directly,
        # but the wrapper makes freeze/unfreeze and param iteration clean.
        self.backbone = ResNet50FeatureExtractor(_full_model)

        if freeze_backbone:
            self.freeze_backbone()
            print("[Backbone] Frozen — transformer + heads training only.")

        # ── 3. 2048 → hidden_dim projection ─────────────────────────
        # ResNet50 layer4 outputs 2048 channels (Bottleneck expansion=4,
        # base_channels=512 → 512*4 = 2048). This projects to transformer dim.
        self.conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # ── 4. Transformer ──────────────────────────────────────────
        self.transformer = nn.Transformer(
            d_model            = hidden_dim,
            nhead              = nheads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            batch_first        = True,
            dropout            = 0.1,
        )

        # ── 5. Prediction heads ─────────────────────────────────────
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)  # +1 = no-object
        self.linear_bbox  = nn.Linear(hidden_dim, 4)

        # ── 6. Queries & normalisation ──────────────────────────────
        self.num_queries = num_queries
        self.query_pos   = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.norm_src    = nn.LayerNorm(hidden_dim)
        self.norm_tgt    = nn.LayerNorm(hidden_dim)

        # Initialise only the new layers (not the pretrained backbone)
        self._init_new_layers()

    # ── Freeze / unfreeze helpers ────────────────────────────────────

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        print("[Backbone] Unfrozen — full end-to-end fine-tuning active.")

    # ── Weight init for non-backbone layers ──────────────────────────

    def _init_new_layers(self):
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.linear_class.weight)
        nn.init.zeros_(self.linear_class.bias)
        nn.init.xavier_uniform_(self.linear_bbox.weight)
        nn.init.zeros_(self.linear_bbox.bias)
        nn.init.normal_(self.query_pos, mean=0.0, std=0.02)

    # ── Checkpoint loading ───────────────────────────────────────────

    def load_pretrained(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd   = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[DETR] Loaded checkpoint: {checkpoint_path}")
        if missing:    print(f"       Missing keys    : {missing}")
        if unexpected: print(f"       Unexpected keys : {unexpected}")

    # ── Parameter summary ────────────────────────────────────────────

    def param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        print(f"\n{'─'*45}")
        print(f"  Total params     : {total:>12,}")
        print(f"  Trainable params : {trainable:>12,}")
        print(f"  Frozen params    : {frozen:>12,}")
        print(f"  Backbone out dim : 2048 (ResNet50 layer4)")
        print(f"  Hidden dim       : {sum(p.numel() for p in self.conv.parameters()):>12,} (conv projection)")
        print(f"{'─'*45}\n")

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, inputs):
        # ① Backbone → spatial feature maps  (B, 2048, Hf, Wf)
        x = self.backbone(inputs)

        # ② Project 2048 → hidden_dim         (B, hidden_dim, Hf, Wf)
        feat = self.conv(x)
        bsz, d_model, Hf, Wf = feat.shape

        # ③ Flatten spatial dims              (B, Hf*Wf, hidden_dim)
        src = feat.flatten(2).permute(0, 2, 1)

        # ④ 2D sine-cos positional encoding   (1, Hf*Wf, hidden_dim)
        pos = build_2d_sincos_position_embedding(
            Hf, Wf, d_model, device=feat.device)
        src = self.norm_src(src + pos)

        # ⑤ Decoder queries
        tgt       = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
        query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt       = self.norm_tgt(tgt + query_pos)

        # ⑥ Transformer                       (B, num_queries, hidden_dim)
        hs = self.transformer(src=src, tgt=tgt)

        # ⑦ Prediction heads
        return {
            "pred_logits": self.linear_class(hs),           # (B, Q, num_classes+1)
            "pred_boxes":  self.linear_bbox(hs).sigmoid(),  # (B, Q, 4) in [0,1]
        }


# ─────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = DETR(
        num_classes    = 17,
        backbone_ckpt  = r"D:\SignDETR\weights\resnet-weights\resnet_epoch_250.pt",
        freeze_backbone= True,
    )
    model.param_summary()
    summary(model, input_size=(4, 3, 224, 224))