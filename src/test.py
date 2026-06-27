import argparse
import json
import logging
import warnings
from pathlib import Path
import pandas as pd
 
import joblib
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
 

# ──────────────────────────────────────────────
# Step 1 — Load embeddings and labels
# ──────────────────────────────────────────────
 
def load_data(preprocessed_dir: Path):
    """
    Load all embeddings and label vectors from the manifest.
    Returns:
        X         : np.ndarray of shape (n_scans, 768)
        y         : np.ndarray of shape (n_scans, 50)
        case_ids  : list of str
    """
    manifest_path = preprocessed_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
 
    log.info(f"Loading data for {len(manifest)} scans")
 
    embeddings = []
    labels     = []
    case_ids   = []
 
    for entry in manifest:
        case_id    = entry["case_id"]
        emb_path   = Path(entry["embedding_path"])
        label_path = Path(entry["label_path"])
 
        if not emb_path.exists():
            log.warning(f"Missing embedding for {case_id}, skipping")
            continue
        if not label_path.exists():
            log.warning(f"Missing label for {case_id}, skipping")
            continue
 
        embeddings.append(np.load(str(emb_path)))     # (768,)
        labels.append(np.load(str(label_path)))        # (50,)
 
    X = np.stack(embeddings, axis=0)   # (n_scans, 768)
    y = np.stack(labels,     axis=0)   # (n_scans, 50)
 
    log.info(f"X shape: {X.shape}  y shape: {y.shape}")
    log.info(f"Positive rate per class (min/mean/max): "
             f"{y.mean(axis=0).min():.3f} / "
             f"{y.mean(axis=0).mean():.3f} / "
             f"{y.mean(axis=0).max():.3f} / "
             f"Classes with zero positives: {np.where(y.sum(axis=0) == 0)[0].tolist()}")
 
    return X, y

X, y = load_data(Path("data/images_preprocessed"))

y = pd.DataFrame(y)

y.to_csv("y.csv")