#!/usr/bin/env python3
"""
multiclass_model.py
===================
v2 phase-classifier model: per-frame attention-pool over SAM2-FPN spatial
features → BiGRU over time → 11-class softmax.

Why this architecture (vs v1's global-pool + linear):
  - The v1 failure mode on OOD recordings is uniform feature-space shift
    (HUD pillarbox lowers avg-pool channels, sharper sensor raises max-pool
    channels). A global pool has no spatial selectivity, so the classifier
    cannot ignore the HUD region. An attention pool can learn to weight
    surgical-field tokens and suppress HUD tokens.
  - v1 is per-frame i.i.d.; a single rogue frame collapses the Viterbi
    timeline. A BiGRU smooths in feature space (a strictly stronger smoother
    than the per-class rolling mean used in v1 post-processing).

Modules
-------
- AttnPool: per-FPN-level pool. 1×1 conv projects to embed_dim → flatten
  spatial tokens → LayerNorm → single learned query × multi-head
  cross-attention → embed_dim vector per frame.
- AttnPoolBiGRU: two AttnPools (one per FPN level), concat to 2·embed_dim,
  BiGRU (1 layer, bidirectional), linear head → NUM_CLASSES logits.

The Viterbi state machine, class names, and 11-class layout are unchanged
from v1 — this module only emits per-frame log-softmax for the existing
decoder.

The model exposes pool_only() and gru_and_head() separately so the
validation script can do test-time burn-in normalisation on pooled features
BEFORE the GRU sees them (the burn-in's purpose is to absorb OOD distribution
shift in the pool output, not the GRU's recurrent state).

Usage
-----
  model = AttnPoolBiGRU(fpn1_channels=64, fpn2_channels=256)
  feats1 = torch.randn(B, T, 64, 16, 16)
  feats2 = torch.randn(B, T, 256, 8, 8)
  logits = model(feats1, feats2)            # (B, T, 11)
  # Or split for TT-norm:
  pooled = model.pool_only(feats1, feats2)  # (B, T, 2*embed_dim)
  pooled = (pooled - mu) / sigma            # burn-in normalisation
  logits = model.gru_and_head(pooled)       # (B, T, 11)
"""

import torch
import torch.nn as nn

NUM_CLASSES = 11


class AttnPool(nn.Module):
    """Attention-pool a spatial feature map (B, C, H, W) → (B, embed_dim).

    A 1×1 conv projects C channels to embed_dim so the learnable query and
    the spatial tokens live in the same space, then a single learned query
    cross-attends over all H·W spatial tokens. The output is the attended
    vector — a spatially-weighted summary that can suppress HUD regions
    once trained.
    """

    def __init__(self, in_channels: int, embed_dim: int = 64,
                 num_heads: int = 2):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim={embed_dim} must divide num_heads={num_heads}"
        self.proj  = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm  = nn.LayerNorm(embed_dim)
        # Single learned query, broadcast over the batch in forward.
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                            batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, embed_dim)."""
        x = self.proj(x)                       # (B, E, H, W)
        B, E, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, E)
        tokens = self.norm(tokens)
        q = self.query.expand(B, -1, -1)       # (B, 1, E)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return out.squeeze(1)                  # (B, E)


class AttnPoolBiGRU(nn.Module):
    """v2 phase classifier.

    Inputs:
        feats_fpn1 : (B, T, C1, H1, W1) — fpn[1] downsampled (default
                     C1=64, H1=W1=16 in extract_multiclass_features_v2.py).
        feats_fpn2 : (B, T, C2, H2, W2) — fpn[2] downsampled (default
                     C2=256, H2=W2=8).

    Output:
        logits     : (B, T, NUM_CLASSES) per-frame class logits.
    """

    def __init__(self,
                 fpn1_channels: int = 64,
                 fpn2_channels: int = 256,
                 embed_dim:     int = 64,
                 gru_hidden:    int = 64,
                 num_heads:     int = 2,
                 num_classes:   int = NUM_CLASSES,
                 dropout:       float = 0.2):
        super().__init__()
        self.embed_dim  = embed_dim
        self.gru_hidden = gru_hidden
        self.feat_dim   = embed_dim * 2          # concat of two pools

        self.pool1 = AttnPool(fpn1_channels, embed_dim, num_heads)
        self.pool2 = AttnPool(fpn2_channels, embed_dim, num_heads)

        self.gru = nn.GRU(
            input_size=self.feat_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(gru_hidden * 2, num_classes)

    def pool_only(self,
                  feats_fpn1: torch.Tensor,
                  feats_fpn2: torch.Tensor) -> torch.Tensor:
        """Forward through the two AttnPools and concat. Useful for
        test-time burn-in normalisation, which fits μ/σ on the pooled
        features before the GRU sees them.

        Returns: (B, T, 2*embed_dim).
        """
        B, T = feats_fpn1.shape[:2]
        f1 = feats_fpn1.reshape(B * T, *feats_fpn1.shape[2:])
        f2 = feats_fpn2.reshape(B * T, *feats_fpn2.shape[2:])
        z1 = self.pool1(f1)
        z2 = self.pool2(f2)
        z  = torch.cat([z1, z2], dim=-1)         # (B*T, 2E)
        return z.view(B, T, -1)

    def gru_and_head(self, z: torch.Tensor) -> torch.Tensor:
        """Run BiGRU + classifier on (already-pooled, optionally normalised)
        per-frame features.

        Args:
            z: (B, T, 2*embed_dim) — output of pool_only(), optionally
               normalised by test-time burn-in.

        Returns:
            (B, T, num_classes) logits.
        """
        z, _ = self.gru(z)
        z = self.dropout(z)
        return self.head(z)

    def forward(self,
                feats_fpn1: torch.Tensor,
                feats_fpn2: torch.Tensor) -> torch.Tensor:
        z = self.pool_only(feats_fpn1, feats_fpn2)
        return self.gru_and_head(z)

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "pool1": sum(p.numel() for p in self.pool1.parameters()),
            "pool2": sum(p.numel() for p in self.pool2.parameters()),
            "gru":   sum(p.numel() for p in self.gru.parameters()),
            "head":  sum(p.numel() for p in self.head.parameters()),
            "total": total,
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Match the v2 extractor's defaults: Hiera-S, fpn[1]=64ch@16x16, fpn[2]=256ch@8x8.
    model = AttnPoolBiGRU(fpn1_channels=64, fpn2_channels=256,
                          embed_dim=64, gru_hidden=64,
                          num_heads=2, num_classes=11)
    print("Param counts:")
    for k, v in model.count_params().items():
        print(f"  {k:<6} {v:>10,}")

    B, T = 2, 64
    f1 = torch.randn(B, T, 64,  16, 16)
    f2 = torch.randn(B, T, 256,  8,  8)
    logits = model(f1, f2)
    print(f"\nforward(): in1={tuple(f1.shape)}  in2={tuple(f2.shape)}  "
          f"out={tuple(logits.shape)}")
    assert logits.shape == (B, T, 11)

    pooled = model.pool_only(f1, f2)
    out2   = model.gru_and_head(pooled)
    print(f"pool_only(): {tuple(pooled.shape)}  → gru_and_head(): "
          f"{tuple(out2.shape)}")
    assert pooled.shape == (B, T, 128)
    assert out2.shape   == (B, T, 11)

    # Verify split path is equivalent to forward()
    torch.manual_seed(0)
    model.eval()
    with torch.no_grad():
        a = model(f1, f2)
        b = model.gru_and_head(model.pool_only(f1, f2))
    assert torch.allclose(a, b, atol=1e-6), "split path diverges from forward()"
    print("split path matches forward()")
    print("\nsmoke test OK")
