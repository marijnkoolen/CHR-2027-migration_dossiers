"""Sequence-context model: a bidirectional Transformer over a PDF's ordered
page embeddings, used to predict Start page (document-boundary detection)
and, using the resulting document segments, Document type / Layout Type
Classification / Functional Categories.

Design:
    1. A shared PageEmbedder (see models.py) turns each page image into one
       embedding - computed once per page, reused by every head below.
    2. A sinusoidal positional encoding is added before the encoder. Plain
       nn.TransformerEncoder has no built-in notion of sequence order -
       without this, self-attention is permutation-invariant and can only
       tell pages apart by content, not position, which starves the model
       of exactly the "this page comes right after that one" signal that
       page-order structure (and therefore Start page and document
       segmentation) depends on.
    3. A Transformer encoder (non-causal: at inference time the whole PDF is
       available, so every page can attend to every other page) turns the
       raw per-page embeddings into contextualized states.
    4. Start-page head: a page is a document boundary if its contextualized
       state looks different from the previous page's, so the head sees
       both the state itself and its difference from the previous page's
       state - an explicit "discontinuity" feature, not just left to
       self-attention to discover.
    5. Segmentation: pages are grouped into runs between consecutive start
       pages (every sequence's first real page always starts a segment).
       During training this uses the ground-truth Start page labels
       ("teacher forcing"), so the type/layout/functional heads learn from
       correct segments; at inference it uses the model's own start-page
       predictions.
    6. Document-level heads: each segment's contextualized states are
       mean-pooled into one vector, broadcast back to every page in that
       segment, and concatenated with the page's own state before
       classifying Document type / Layout / Functional. Pages in the same
       document therefore share context, not just their own image.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def compute_segment_ids(start_bin: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """start_bin, padding_mask: (B, T) bool. Returns (B, T) long segment ids,
    0-indexed per row, -1 for padded positions. Real pages are always
    left-aligned (padding, if any, comes after), so position 0 is always a
    real page and is always forced to begin a segment, regardless of its
    start-page label/prediction."""
    start_bin = start_bin.clone()
    start_bin[:, 0] = True
    seg_ids = torch.cumsum(start_bin.long(), dim=1) - 1
    seg_ids = seg_ids.masked_fill(padding_mask, -1)
    return seg_ids


def segment_mean_pool(states: torch.Tensor, seg_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pools `states` (B, T, D) within each segment and broadcasts the
    pooled vector back to every page of that segment. Implemented as a
    per-row loop (batches here are a handful of whole PDFs, tens of pages
    each - this is not a bottleneck) so it needs no extra dependency.

    Accumulates in float32 regardless of `states`' own dtype: a segment can
    be dozens of pages long, and summing that many terms in bf16 (~3 decimal
    digits) loses precision fast - then casts back to the original dtype."""
    B, T, D = states.shape
    orig_dtype = states.dtype
    states = states.float()
    out = torch.zeros_like(states)
    for b in range(B):
        valid = ~padding_mask[b]
        ids = seg_ids[b][valid]
        vecs = states[b][valid]
        if ids.numel() == 0:
            continue
        n_segs = int(ids.max().item()) + 1
        sums = torch.zeros(n_segs, D, device=states.device, dtype=states.dtype)
        counts = torch.zeros(n_segs, device=states.device, dtype=states.dtype)
        sums.index_add_(0, ids, vecs)
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=states.dtype))
        means = sums / counts.clamp(min=1).unsqueeze(-1)
        out[b][valid] = means[ids]
    return out.to(orig_dtype)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal position encoding (Vaswani et al.), recomputed on
    every forward call rather than cached in a fixed-size buffer, so it has
    no built-in maximum sequence length - real PDFs already range from a
    handful of pages to 80+, and there's no reason to assume future
    documents fit under some hardcoded cap."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, D = x.shape
        position = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe = torch.zeros(T, D, device=x.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0).to(dtype=x.dtype)


class SequenceContextModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_doctype: int,
        num_layout: int,
        num_functional: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.positional_encoding = SinusoidalPositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
        )
        # enable_nested_tensor=False: PyTorch's padding-mask fast path (used
        # under torch.no_grad(), i.e. at eval time) converts to a nested
        # tensor internally, which isn't implemented on MPS as of this
        # writing. Disabling it forces the standard path everywhere so
        # train/eval behave identically across devices.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        self.start_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2), nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1)
        )
        doc_in = embed_dim * 2

        def doc_head(n_out):
            return nn.Sequential(nn.LayerNorm(doc_in), nn.Linear(doc_in, embed_dim), nn.GELU(), nn.Linear(embed_dim, n_out))

        self.doctype_head = doc_head(num_doctype)
        self.layout_head = doc_head(num_layout)
        self.functional_head = doc_head(num_functional)

    def forward(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
        true_start_page: torch.Tensor | None = None,
    ) -> dict:
        """embeddings: (B, T, D) raw per-page embeddings.
        padding_mask: (B, T) bool, True at padded positions.
        true_start_page: (B, T) bool/float, ground truth for teacher forcing
            during training; pass None to use the model's own predictions
            (inference / eval).
        """
        embeddings = self.positional_encoding(embeddings)
        context = self.encoder(embeddings, src_key_padding_mask=padding_mask)

        # Boundary detection depends on the *difference* between two
        # adjacent, usually-similar embeddings - exactly the kind of
        # small-magnitude subtraction that loses most of its relative
        # precision under bf16 autocast (~3 decimal digits: two nearly
        # equal values can subtract down to something that's mostly
        # rounding error). Force this computation, and the head that
        # consumes it, to run in float32 regardless of the surrounding
        # autocast context - a standard mixed-precision-training safeguard
        # for numerically delicate ops, not just a stylistic cast.
        with torch.autocast(device_type=context.device.type, enabled=False):
            context_fp32 = context.float()
            prev_fp32 = torch.zeros_like(context_fp32)
            prev_fp32[:, 1:] = context_fp32[:, :-1]
            boundary_feat = context_fp32 - prev_fp32
            start_logits = self.start_head(torch.cat([context_fp32, boundary_feat], dim=-1)).squeeze(-1)

        if true_start_page is not None:
            start_bin = true_start_page.bool()
        else:
            start_bin = torch.sigmoid(start_logits) > 0.5

        seg_ids = compute_segment_ids(start_bin, padding_mask)
        seg_vec = segment_mean_pool(context, seg_ids, padding_mask)
        doc_feat = torch.cat([context, seg_vec], dim=-1)

        return {
            "start_logits": start_logits,
            "doctype_logits": self.doctype_head(doc_feat),
            "layout_logits": self.layout_head(doc_feat),
            "functional_logits": self.functional_head(doc_feat),
            "segment_ids": seg_ids,
        }
