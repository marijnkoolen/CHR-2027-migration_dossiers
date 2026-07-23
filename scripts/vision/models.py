"""Model builders shared across the training scripts.

Both single-image scenarios (train_efficient.py, train_quality.py) and the
sequence-context scenario (train_sequence.py) use the same recipe for
turning an image into a feature vector - a pretrained ViT-style backbone
loaded via `transformers.AutoModel`, [CLS]-token pooled - just with a
different backbone size and freeze/fine-tune policy. This works for DINOv2
checkpoints (Dinov2Model) and DiT checkpoints (BeitModel, since DiT reuses
the BEiT architecture) without needing separate code paths. `PageEmbedder`
holds that shared logic; `BackboneClassifier` adds a plain classification
head on top of it for the single-image scripts, and `sequence_model.py`
attaches a page-embedder to a sequence-context head instead.

`TextEmbedder` is the same idea for a page's transcribed text (from
PageXML), and `MultimodalPageEmbedder` late-fuses the two by concatenating
their embeddings - see train_multimodal.py for how it's used.
"""

from __future__ import annotations

import inspect

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def trainable_parameter_summary(module: nn.Module) -> str:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return f"{trainable:,} / {total:,} parameters trainable ({100 * trainable / total:.1f}%)"


def _find_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Locate the list of transformer blocks regardless of backbone family
    (Dinov2Model: model.encoder.layer; BeitModel/DiT: model.layers;
    XLMRobertaModel and friends: model.encoder.layer)."""
    for path in ("encoder.layer", "layers", "encoder.layers", "blocks"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"Could not find transformer blocks on {type(model).__name__}")


def _attn_implementation_kwargs(device: torch.device | None) -> dict:
    """MPS's scaled_dot_product_attention kernel doesn't support dropout (as
    of this PyTorch version) - hit during training, since attention dropout
    is only active in train() mode. HF's default 'sdpa' attention
    implementation uses that kernel and so errors there; fall back to the
    slower but backend-agnostic 'eager' implementation on MPS only, leaving
    the faster/more memory-efficient SDPA path on CUDA/CPU."""
    if device is not None and device.type == "mps":
        return {"attn_implementation": "eager"}
    return {}


def _freeze_all_but_last_n(
    backbone: nn.Module, unfreeze_last_n: int, gradient_checkpointing: bool = False
) -> None:
    for p in backbone.parameters():
        p.requires_grad = False
    if unfreeze_last_n > 0:
        blocks = _find_transformer_blocks(backbone)
        for block in blocks[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True
        # Only useful (and only takes effect) when something is actually
        # unfrozen: a fully frozen backbone builds no backward graph through
        # itself at all (no output here requires grad, so autograd already
        # skips retaining its activations) - checkpointing has nothing to
        # save memory on there, it would just recompute forward for nothing.
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable()


class PageEmbedder(nn.Module):
    """Pretrained ViT-style backbone -> single embedding per image, with a
    configurable number of trailing transformer blocks left trainable
    (0 = frozen, linear-probe style).

    project_to: optionally project the backbone's native embedding down to a
    smaller size. Matters most for sequence mode: SequenceContextModel's
    heads scale with embed_dim (doc_in = embed_dim*2), so a 1024-dim backbone
    (DiT-large) gives it ~7x the parameters a 384-dim one (DINOv2-small)
    would - badly overparameterized against a PDF-level training set that's
    typically just a few dozen documents, which in practice collapses
    training onto the trivial "predict the label's marginal frequency,
    ignore the page" solution regardless of learning rate or model depth.
    Projecting down to something in DINOv2-small's range (e.g. 384) before
    the sequence model fixes this without giving up the larger backbone's
    features entirely."""

    def __init__(self, backbone_name: str, unfreeze_last_n_blocks: int = 0, device: torch.device | None = None,
                 gradient_checkpointing: bool = False, project_to: int | None = None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name, **_attn_implementation_kwargs(device))
        backbone_dim = self.backbone.config.hidden_size
        # Some backbones (Beit/DiT) need an explicit flag to interpolate their
        # position embeddings for input sizes other than the pretraining
        # resolution; others (Dinov2) handle arbitrary sizes automatically.
        self._supports_pos_interp = "interpolate_pos_encoding" in inspect.signature(
            self.backbone.forward
        ).parameters
        # Beit-family checkpoints (DiT included) trained with
        # use_mean_pooling=True read out via a separate pooler (LayerNorm
        # over mean-pooled patch tokens, `pooler_output`) rather than the
        # [CLS] token - for those, last_hidden_state[:, 0] is essentially an
        # untrained, unstable readout (seen empirically: std ~70 vs ~1 for
        # the pooler, occasionally spiking into the thousands and blowing up
        # downstream LayerNorms/losses). Dinov2 and plain ViT checkpoints
        # don't set this flag and use [CLS] as intended.
        self._use_pooler = bool(getattr(self.backbone.config, "use_mean_pooling", False))
        _freeze_all_but_last_n(self.backbone, unfreeze_last_n_blocks, gradient_checkpointing)

        if project_to:
            self.projection = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, project_to), nn.GELU())
            self.embed_dim = project_to
        else:
            self.projection = nn.Identity()
            self.embed_dim = backbone_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        kwargs = {"interpolate_pos_encoding": True} if self._supports_pos_interp else {}
        outputs = self.backbone(pixel_values=pixel_values, **kwargs)
        cls = outputs.pooler_output if self._use_pooler else outputs.last_hidden_state[:, 0]
        return self.projection(cls)


class BackboneClassifier(nn.Module):
    """A PageEmbedder + linear head, for plain single-image classification."""

    def __init__(self, backbone_name: str, num_classes: int, unfreeze_last_n_blocks: int = 0,
                 device: torch.device | None = None, gradient_checkpointing: bool = False,
                 project_to: int | None = None):
        super().__init__()
        self.embedder = PageEmbedder(backbone_name, unfreeze_last_n_blocks, device=device,
                                      gradient_checkpointing=gradient_checkpointing, project_to=project_to)
        self.head = nn.Sequential(nn.LayerNorm(self.embedder.embed_dim), nn.Linear(self.embedder.embed_dim, num_classes))

    @property
    def backbone(self):
        return self.embedder.backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedder(pixel_values))


class TextEmbedder(nn.Module):
    """Pretrained multilingual transformer -> single embedding per page's
    transcribed text, mean-pooled over non-padding tokens (a better
    off-the-shelf sentence representation than the [CLS]/<s> token when the
    backbone is frozen or only lightly fine-tuned)."""

    def __init__(self, backbone_name: str = "xlm-roberta-base", unfreeze_last_n_layers: int = 0,
                 max_length: int = 256, device: torch.device | None = None, gradient_checkpointing: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name, **_attn_implementation_kwargs(device))
        self.embed_dim = self.backbone.config.hidden_size
        self.max_length = max_length
        _freeze_all_but_last_n(self.backbone, unfreeze_last_n_layers, gradient_checkpointing)

    def tokenize(self, texts: list[str]):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts


class MultimodalPageEmbedder(nn.Module):
    """Late fusion of PageEmbedder (image) and TextEmbedder (transcribed
    text): each modality is embedded independently and the two vectors are
    concatenated (optionally projected back down to a chosen size), so this
    exposes the same fixed-size `embed_dim` output as a plain PageEmbedder
    and can be dropped in wherever one is expected."""

    def __init__(
        self,
        image_backbone: str,
        text_backbone: str = "xlm-roberta-base",
        unfreeze_image_blocks: int = 0,
        unfreeze_text_layers: int = 0,
        max_text_length: int = 256,
        project_to: int | None = None,
        device: torch.device | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.image_embedder = PageEmbedder(image_backbone, unfreeze_image_blocks, device=device,
                                            gradient_checkpointing=gradient_checkpointing)
        self.text_embedder = TextEmbedder(text_backbone, unfreeze_text_layers, max_text_length, device=device,
                                           gradient_checkpointing=gradient_checkpointing)

        combined_dim = self.image_embedder.embed_dim + self.text_embedder.embed_dim
        if project_to:
            self.projection = nn.Sequential(nn.LayerNorm(combined_dim), nn.Linear(combined_dim, project_to), nn.GELU())
            self.embed_dim = project_to
        else:
            self.projection = nn.Identity()
            self.embed_dim = combined_dim

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        image_embed = self.image_embedder(pixel_values)
        text_embed = self.text_embedder(input_ids, attention_mask)
        return self.projection(torch.cat([image_embed, text_embed], dim=-1))


class MultimodalBackboneClassifier(nn.Module):
    """A MultimodalPageEmbedder + linear head."""

    def __init__(
        self,
        image_backbone: str,
        num_classes: int,
        text_backbone: str = "xlm-roberta-base",
        unfreeze_image_blocks: int = 0,
        unfreeze_text_layers: int = 0,
        max_text_length: int = 256,
        project_to: int | None = None,
        device: torch.device | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.embedder = MultimodalPageEmbedder(
            image_backbone, text_backbone, unfreeze_image_blocks, unfreeze_text_layers, max_text_length, project_to,
            device=device, gradient_checkpointing=gradient_checkpointing,
        )
        self.head = nn.Sequential(nn.LayerNorm(self.embedder.embed_dim), nn.Linear(self.embedder.embed_dim, num_classes))

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedder(pixel_values, input_ids, attention_mask))
