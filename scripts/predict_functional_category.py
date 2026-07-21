"""
predict_functional_category.py
================================
Train Late-Fusion (EfficientNet-B0 FT + BERT FT) on the 65 labeled dossiers
from merged_annotations.tsv, then predict functional_category for every page
in the remaining unlabeled dossiers.

Inputs
------
  data/annotations/merged_annotations.tsv    – labeled pages (training data)
  image-per-page/<dossier>/               – PNG images per page
  extract-text-per-page/<dossier>/page/   – PAGE-XML text files per page

Output
------
  data/predictions/functional_category_predictions.csv
      dossier, page_number, predicted_functional_category,
      confidence, prob_<class> × 7
"""

import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent
IMG_ROOT  = WORKSPACE / "data" / "image-per-page"
TEXT_ROOT = WORKSPACE / "data" / "text-per-page"
ANN_CSV   = WORKSPACE / "data" / "labels" / "merged_annotations.tsv"
OUT_DIR   = WORKSPACE / "data" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = WORKSPACE / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Administrative & Internal Processing Documents",   # 0
    "Application Documents",                            # 1
    "Decision Documents",                               # 2
    "Medical & Health Documents",                       # 3
    "Other",                                            # 4
    "Qualification & Employment Proof",                 # 5
    "Security & Political Screening Documents",         # 6
]
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

BERT_MODEL   = "bert-base-uncased"
RANDOM_SEED  = 42
MAX_TEXT_LEN = 256
IMG_BATCH    = 16
TXT_BATCH    = 8
PRED_BATCH   = 32

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)
print(f"Device: {DEVICE}")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── Text extraction (PAGE-XML) ─────────────────────────────────────────────────
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

def read_page_xml(path: Path) -> str:
    """Extract all Unicode text from a PAGE-XML file in reading order."""
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
        parts = []
        for te in root.iter(f"{{{PAGE_NS}}}Unicode"):
            if te.text:
                parts.append(te.text.strip())
        return " ".join(parts)
    except Exception:
        return ""


# ── Path helpers ───────────────────────────────────────────────────────────────
def img_path(dossier_dir: str, page_num: int) -> Path:
    return IMG_ROOT / dossier_dir / f"{dossier_dir}_page_{int(page_num):04d}.png"

def txt_path(dossier_dir: str, page_num: int) -> Path:
    return TEXT_ROOT / dossier_dir / "page" / f"page_{int(page_num):04d}.xml"


# ── Labeled data loader ────────────────────────────────────────────────────────
def load_labeled_data() -> pd.DataFrame:
    df = pd.read_csv(ANN_CSV, sep='\t')
    df["dossier_dir"] = df["dossier_name"].str.replace(r"\.pdf$", "", regex=True)
    col_map = {
        'page number': 'page_num',
        'Document type': 'doc_type', 
        'Functional Categories': 'functional_category',
    }
    df = df.rename(columns=col_map)
    df["page_num"] = pd.to_numeric(df["page_num"], errors="coerce")
    df = df.dropna(subset=["page_num", "functional_category"])
    df["page_num"] = df["page_num"].astype(int)
    df = df[df["functional_category"].isin(CLASS_TO_IDX)].copy()
    df["label"]    = df["functional_category"].map(CLASS_TO_IDX).astype(int)
    df["img_path"] = df.apply(lambda r: img_path(r["dossier_dir"], r["page_num"]), axis=1)
    df["txt_path"] = df.apply(lambda r: txt_path(r["dossier_dir"], r["page_num"]), axis=1)
    mask = df["img_path"].map(lambda p: p.exists())
    missing = (~mask).sum()
    if missing:
        print(f"  Warning: {missing} labeled pages have no image – skipping.")
    df = df[mask].reset_index(drop=True)
    print(f"Labeled pages: {len(df)}  |  dossiers: {df['dossier_dir'].nunique()}")
    return df


# ── Unlabeled pages inventory ──────────────────────────────────────────────────
def build_unlabeled_pages(labeled_dossiers: set) -> pd.DataFrame:
    rows = []
    all_dirs = sorted(d.name for d in IMG_ROOT.iterdir() if d.is_dir())
    print(f"Total dossier directories: {len(all_dirs)}")
    skipped = 0
    for dossier_dir in all_dirs:
        if dossier_dir in labeled_dossiers:
            skipped += 1
            continue
        for img_file in sorted((IMG_ROOT / dossier_dir).glob(f"{dossier_dir}_page_*.png")):
            try:
                page_num = int(img_file.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            rows.append({
                "dossier_dir": dossier_dir,
                "page_num":    page_num,
                "img_path":    img_file,
                "txt_path":    txt_path(dossier_dir, page_num),
            })
    df = pd.DataFrame(rows)
    print(f"Skipped {skipped} labeled dossiers. "
          f"Unlabeled pages: {len(df)} across {df['dossier_dir'].nunique()} dossiers.")
    return df


# ── Datasets ───────────────────────────────────────────────────────────────────
class PageImageDataset(Dataset):
    def __init__(self, paths: list, labels: list, transform):
        self.paths     = [str(p) for p in paths]
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, paths: list, labels: list, tokenizer, max_length: int = MAX_TEXT_LEN):
        print(f"    Reading {len(paths)} XML text files…")
        self.texts     = [read_page_xml(p) for p in paths]
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.labels[i], dtype=torch.long),
        )


# ── Utilities ──────────────────────────────────────────────────────────────────
def class_weights_tensor(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    w = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)


train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── EfficientNet-B0 fine-tuned ─────────────────────────────────────────────────
def build_efficientnet() -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = efficientnet_b0(weights=weights)
    for name, p in model.named_parameters():
        if "features.0" in name or "features.1" in name:
            p.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model.to(DEVICE)


def train_image_model(model: nn.Module,
                      tr_loader: DataLoader,
                      va_loader: DataLoader,
                      y_tr: np.ndarray,
                      y_va: np.ndarray,
                      n_epochs: int = 15,
                      lr: float = 5e-5) -> nn.Module:
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor(y_tr))
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_f1, best_state = -1.0, None

    for epoch in range(1, n_epochs + 1):
        model.train()
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(imgs), labels).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, _ in va_loader:
                preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().tolist())
        f1 = f1_score(y_va, preds, average="macro", zero_division=0)
        print(f"  Epoch {epoch:2d}  val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  ✓ Best EfficientNet val_macro_f1={best_f1:.4f}")
    return model


@torch.no_grad()
def predict_image_model(model: nn.Module, loader: DataLoader):
    model.eval()
    preds, probs_list = [], []
    for imgs, _ in loader:
        logits = model(imgs.to(DEVICE))
        probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        preds.extend(logits.argmax(1).cpu().tolist())
    return np.array(preds), np.vstack(probs_list)


# ── BERT fine-tuned ────────────────────────────────────────────────────────────
class BERTClassifier(nn.Module):
    def __init__(self, model_name: str = BERT_MODEL,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


def train_bert_model(model: nn.Module,
                     tr_loader: DataLoader,
                     va_loader: DataLoader,
                     y_tr: np.ndarray,
                     y_va: np.ndarray,
                     n_epochs: int = 10,
                     lr: float = 2e-5) -> nn.Module:
    cw   = class_weights_tensor(y_tr)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch  = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    best_f1, best_state = -1.0, None

    for epoch in range(1, n_epochs + 1):
        model.train()
        for ids, mask, yb in tr_loader:
            ids, mask, yb = ids.to(DEVICE), mask.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(ids, mask), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for ids, mask, _ in va_loader:
                preds.extend(
                    model(ids.to(DEVICE), mask.to(DEVICE)).argmax(1).cpu().tolist()
                )
        f1 = f1_score(y_va, preds, average="macro", zero_division=0)
        print(f"  Epoch {epoch:2d}  val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  ✓ Best BERT val_macro_f1={best_f1:.4f}")
    return model


@torch.no_grad()
def predict_bert_model(model: nn.Module, loader: DataLoader):
    model.eval()
    preds, probs_list = [], []
    for ids, mask, _ in loader:
        logits = model(ids.to(DEVICE), mask.to(DEVICE))
        probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        preds.extend(logits.argmax(1).cpu().tolist())
    return np.array(preds), np.vstack(probs_list)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Load labeled data
    print("\n" + "="*70)
    print("Step 1: Loading labeled data")
    print("="*70)
    labeled_df = load_labeled_data()
    labeled_dossiers = set(labeled_df["dossier_dir"].unique())

    # 2. Dossier-level 80/20 train/val split for early stopping
    all_dossiers = labeled_df["dossier_dir"].unique()
    train_dos, val_dos = train_test_split(
        all_dossiers, test_size=0.20, random_state=RANDOM_SEED
    )
    tr_df = labeled_df[labeled_df["dossier_dir"].isin(set(train_dos))].reset_index(drop=True)
    va_df = labeled_df[labeled_df["dossier_dir"].isin(set(val_dos))].reset_index(drop=True)
    y_tr  = tr_df["label"].values.astype(np.int32)
    y_va  = va_df["label"].values.astype(np.int32)
    print(f"Train: {len(tr_df)} pages ({len(train_dos)} dossiers)  "
          f"| Val: {len(va_df)} pages ({len(val_dos)} dossiers)")

    # 3. Build unlabeled pages inventory
    print("\n" + "="*70)
    print("Step 2: Building unlabeled page inventory")
    print("="*70)
    unlabeled_df = build_unlabeled_pages(labeled_dossiers)

    # 4. Train EfficientNet-B0 fine-tuned
    print("\n" + "="*70)
    print("Step 3: Training EfficientNet-B0 (fine-tuned)")
    print("="*70)
    
    tr_img_loader = DataLoader(
        PageImageDataset(tr_df["img_path"].tolist(), y_tr.tolist(), train_tfm),
        batch_size=IMG_BATCH, shuffle=True, num_workers=0,
    )
    
    va_img_loader = DataLoader(
        PageImageDataset(va_df["img_path"].tolist(), y_va.tolist(), eval_tfm),
        batch_size=PRED_BATCH, shuffle=False, num_workers=0,
    )

    eff_model_path = MODEL_DIR / "efficientnet_func_cat.pt"
    eff_model = build_efficientnet()
    if eff_model_path.exists():
        eff_model.load_state_dict(torch.load(eff_model_path))
    else:
        trainable = sum(p.numel() for p in eff_model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")
        eff_model = train_image_model(
            eff_model, tr_img_loader, va_img_loader, y_tr, y_va, n_epochs=15
        )
        torch.save(eff_model.state_dict(), MODEL_DIR / "efficientnet_func_cat.pt")

    # 5. Train BERT fine-tuned
    bert_clf_path = MODEL_DIR / "bert_func_cat.pt"
    print("\n" + "="*70)
    print("Step 4: Training BERT (fine-tuned)")
    print("="*70)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=True)

    print("  Loading train text…")
    tr_text_ds = TextDataset(tr_df["txt_path"].tolist(), y_tr.tolist(), tokenizer)
    print("  Loading val text…")
    va_text_ds = TextDataset(va_df["txt_path"].tolist(), y_va.tolist(), tokenizer)

    tr_text_loader = DataLoader(tr_text_ds, batch_size=TXT_BATCH, shuffle=True,  num_workers=0)
    va_text_loader = DataLoader(va_text_ds, batch_size=16,         shuffle=False, num_workers=0)

    bert_clf = BERTClassifier().to(DEVICE)
    if bert_clf_path.exists():
        bert_clf.load_state_dict(torch.load(bert_clf_path))
    else:
        print(f"  Total params: {sum(p.numel() for p in bert_clf.parameters()):,}")
        bert_clf = train_bert_model(
            bert_clf, tr_text_loader, va_text_loader, y_tr, y_va, n_epochs=10
        )
        torch.save(bert_clf.state_dict(), bert_clf_path)

    # 6. Predict unlabeled pages
    print("\n" + "="*70)
    print(f"Step 5: Predicting {len(unlabeled_df):,} unlabeled pages")
    print("="*70)

    CHUNK = 1024  # pages per chunk to manage memory
    all_results = []
    n_chunks = (len(unlabeled_df) + CHUNK - 1) // CHUNK

    for ci, chunk_start in enumerate(range(0, len(unlabeled_df), CHUNK)):
        chunk_df = unlabeled_df.iloc[chunk_start : chunk_start + CHUNK].reset_index(drop=True)
        print(f"\n  Chunk {ci+1}/{n_chunks}  "
              f"(pages {chunk_start}–{chunk_start+len(chunk_df)-1})")

        dummy_labels = [0] * len(chunk_df)

        # Image inference
        img_ds = PageImageDataset(
            chunk_df["img_path"].tolist(), dummy_labels, eval_tfm
        )
        img_loader = DataLoader(img_ds, batch_size=PRED_BATCH, shuffle=False, num_workers=0)
        _, probs_eff = predict_image_model(eff_model, img_loader)

        # Text inference
        txt_ds = TextDataset(
            chunk_df["txt_path"].tolist(), dummy_labels, tokenizer
        )
        txt_loader = DataLoader(txt_ds, batch_size=PRED_BATCH, shuffle=False, num_workers=0)
        _, probs_bert = predict_bert_model(bert_clf, txt_loader)

        # Late fusion
        probs_late   = (probs_eff + probs_bert) / 2.0
        pred_indices = probs_late.argmax(axis=1)
        confidences  = probs_late.max(axis=1)

        for i in range(len(chunk_df)):
            row = chunk_df.iloc[i]
            result = {
                "dossier":                         row["dossier_dir"],
                "page_number":                     row["page_num"],
                "predicted_functional_category":   CLASS_NAMES[pred_indices[i]],
                "confidence":                      round(float(confidences[i]), 4),
            }
            for ci2, cname in enumerate(CLASS_NAMES):
                col = "prob_" + cname.replace(" & ", "_").replace(" ", "_").replace(",", "")
                result[col] = round(float(probs_late[i, ci2]), 4)
            all_results.append(result)

    # 7. Save
    results_df = pd.DataFrame(all_results)
    out_path   = OUT_DIR / "functional_category_predictions.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n{'='*70}")
    print(f"Saved {len(results_df):,} predictions → {out_path}")

    print("\nPrediction class distribution:")
    dist = results_df["predicted_functional_category"].value_counts()
    for cls, cnt in dist.items():
        print(f"  {cls:<55s} {cnt:6d}  ({cnt/len(results_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
