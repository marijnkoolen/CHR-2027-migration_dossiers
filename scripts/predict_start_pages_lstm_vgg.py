"""
Predict start pages for unannotated dossiers using the best LSTM+VGG-16 model.

The saved checkpoint (outputs/page-classifier-baselines/vgg_feature_lstm.pt) uses:
  - VGG-16 penultimate-FC features (4096-D) as page-level visual embeddings
  - A single-layer unidirectional LSTM (hidden_size=256) operating over the
    full page sequence of each dossier
  - A linear head (256 → 2) producing per-page logits

Usage:
    python predict_start_pages_lstm_vgg.py
    python predict_start_pages_lstm_vgg.py --batch_size 32 --output predictions_lstm_vgg.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE  = Path(__file__).parent
PNG_ROOT   = WORKSPACE / "pdf_pages_png"
TEXT_ROOT  = WORKSPACE / "transcriptions-Qwen-7B-dossiers_3343-markdown"
CKPT_PATH  = WORKSPACE / "outputs" / "page-classifier-baselines" / "vgg_feature_lstm.pt"
LABELS_TSV = WORKSPACE / "dossier_labels_merged_pdf12_stratified.tsv"
OUT_PATH   = WORKSPACE / "predictions_lstm_vgg.tsv"

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)


class LSTMPageClassifier(nn.Module):
    """
    Matches the weights saved in vgg_feature_lstm.pt:
      - lstm: nn.LSTM(input_size=4096, hidden_size=256, num_layers=1, batch_first=True)
      - fc:   nn.Linear(256, 2)
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 256) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        out, _ = self.lstm(x)
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.fc(out)   # (B, T, 2)


# ---------------------------------------------------------------------------
# VGG-16 feature extractor (identical to page_classifier_features.py)
# ---------------------------------------------------------------------------
class VGG16FeatureExtractor:
    def __init__(self, device: torch.device, batch_size: int = 16) -> None:
        self.device     = device
        self.batch_size = batch_size
        weights   = models.VGG16_Weights.IMAGENET1K_V1
        vgg       = models.vgg16(weights=weights).to(device)
        # strip final classification layer → 4096-D penultimate FC
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self._model      = vgg
        self._preprocess = weights.transforms()

    @torch.no_grad()
    def encode_paths(self, paths: list[Path]) -> np.ndarray:
        n = len(paths)
        X = np.empty((n, 4096), dtype=np.float32)
        for start in range(0, n, self.batch_size):
            chunk = paths[start : start + self.batch_size]
            tensors = []
            for p in chunk:
                with Image.open(p).convert("RGB") as im:
                    tensors.append(self._preprocess(im))
            xb   = torch.stack(tensors).to(self.device)
            feat = self._model(xb).cpu().numpy().astype(np.float32)
            X[start : start + len(chunk)] = feat
        return X


def ann_to_trans_name(dossier: str) -> str:
    """Convert PNG-folder dossier name (all dashes) to transcription folder name
    (underscores inside the name part, dashes around prefix/suffix).
    e.g. 'a2478-aalbers-b-w-1452065' → 'a2478-aalbers_b_w-1452065'
    """
    parts = dossier.split("-")
    # parts[0]  = 'a2478'
    # parts[-1] = numeric ID
    # parts[1:-1] = name tokens  →  joined with '_'
    return f"{parts[0]}-{'_'.join(parts[1:-1])}-{parts[-1]}"


def text_path(dossier: str, page_num: int) -> Path:
    trans_name = ann_to_trans_name(dossier)
    return TEXT_ROOT / trans_name / f"{trans_name}_page_{page_num:04d}.markdown.md"


def sorted_pages(dossier_dir: Path) -> list[tuple[int, Path]]:
    """Return (page_num, path) tuples sorted by page number."""
    pages = []
    for p in dossier_dir.glob("*.png"):
        stem = p.stem  # e.g. 'a2478-aalbers-b-w-1452065_page_0003'
        try:
            page_num = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        pages.append((page_num, p))
    return sorted(pages, key=lambda x: x[0])

def main(args: argparse.Namespace) -> None:
    if not LABELS_TSV.exists():
        sys.exit(f"Labels TSV not found: {LABELS_TSV}")

    labels_df    = pd.read_csv(LABELS_TSV, sep="\t")
    annotated_ids = set(labels_df["pdf_id"].unique())
    print(f"Annotated dossiers (excluded): {len(annotated_ids)}")

    # ── Discover unannotated dossiers ────────────────────────────────────────
    all_png_dossiers = sorted(
        [d for d in PNG_ROOT.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    unannotated = [d for d in all_png_dossiers if d.name not in annotated_ids]
    print(f"Total PNG dossiers:       {len(all_png_dossiers)}")
    print(f"Unannotated (to predict): {len(unannotated)}")

    if not unannotated:
        print("Nothing to predict – all dossiers are already annotated.")
        return

    # ── Build VGG extractor ──────────────────────────────────────────────────
    print(f"\nDevice: {DEVICE}")
    print("Loading VGG-16 feature extractor …")
    extractor = VGG16FeatureExtractor(device=DEVICE, batch_size=args.batch_size)

    # ── Load LSTM model ──────────────────────────────────────────────────────
    if not CKPT_PATH.exists():
        sys.exit(f"Checkpoint not found: {CKPT_PATH}")

    print(f"Loading LSTM checkpoint from {CKPT_PATH} …")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    feat_dim   = ckpt.get("feat_dim", 4096)
    hidden_dim = ckpt.get("hidden",   256)

    lstm_model = LSTMPageClassifier(input_dim=feat_dim, hidden_dim=hidden_dim).to(DEVICE)
    lstm_model.load_state_dict(ckpt["state_dict"])
    lstm_model.eval()
    print(f"  architecture: LSTM(input={feat_dim}, hidden={hidden_dim})")
    print(f"  parameters:   {sum(p.numel() for p in lstm_model.parameters()):,}")

    # ── Run inference ────────────────────────────────────────────────────────
    rows: list[dict] = []

    for dossier_dir in tqdm(unannotated, desc="Predicting dossiers", unit="dossier"):
        dossier = dossier_dir.name
        pages   = sorted_pages(dossier_dir)

        if not pages:
            continue

        page_nums = [pn for pn, _ in pages]
        img_paths = [p  for _,  p in pages]

        # extract VGG features → (T, 4096)
        feats = extractor.encode_paths(img_paths)
        feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, 4096)
        lengths = torch.tensor([len(page_nums)])

        with torch.no_grad():
            logits = lstm_model(feats_t, lengths)            # (1, T, 2)
            probs  = torch.softmax(logits, dim=-1)[0, :, 1]  # (T,) probability of start-page
            preds  = logits.argmax(-1)[0]                    # (T,)

        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()

        for page_num, img_path, pred, prob in zip(page_nums, img_paths, preds_np, probs_np):
            trans_name = ann_to_trans_name(dossier)
            txt_path   = text_path(dossier, page_num)
            rows.append({
                "dossier":              dossier,
                "page_num":             page_num,
                "img_path":             str(img_path.relative_to(WORKSPACE)),
                "text_path":            str(txt_path.relative_to(WORKSPACE)) if txt_path.exists() else "",
                "predicted_start_page": "yes" if pred == 1 else "no",
                "start_page_prob":      round(float(prob), 4),
            })

    # ── Save ─────────────────────────────────────────────────────────────────
    result_df = pd.DataFrame(rows)

    # summary
    n_pages  = len(result_df)
    n_starts = (result_df["predicted_start_page"] == "yes").sum()
    print(f"\nPrediction summary:")
    print(f"  Total pages predicted:  {n_pages}")
    print(f"  Predicted start pages:  {n_starts}  ({n_starts/max(n_pages,1)*100:.1f}%)")
    print(f"  Dossiers processed:     {result_df['dossier'].nunique()}")

    out_path = Path(args.output)
    result_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved predictions → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict start pages for unannotated dossiers using LSTM+VGG-16."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="VGG feature extraction batch size (default: 16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUT_PATH),
        help="Path to output TSV file",
    )
    args = parser.parse_args()
    main(args)
