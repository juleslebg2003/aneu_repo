"""
TopAneu 2026 - Task 1
Baseline v0 - Step 2 : Multi-label classification 
 
Loads precomputed embeddings from the manifest, trains one logistic regression
per class using MultilabelStratifiedKFold CV, evaluates per-class and macro
AUROC, then retrains on all data and saves the final model.
 
Input  : preprocessed/manifest.json (with embedding_path and label_path entries)
Output : results/cv_results.json + results/baseline_v0_model.joblib
 
Usage:
    python train_baseline.py \
        --preprocessed_dir /path/to/preprocessed \
        --output_dir       /path/to/results

"""
import json 
from pathlib import Path 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np 
import matplotlib.pyplot as plt 
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EMBEDDING_SHAPE = (768,)
LABEL_SHAPE = (49,)

# ──────────────────────────────────────────────
# Step 1 - Load embeddings and labels 
# ──────────────────────────────────────────────

# Loading embeddings and labels from manifest to match scikit-learn input format requirement
def matrix_creation(manifest_path : Path) -> np.array: 
    X = np.zeros((98, 768))
    y = np.zeros((98, 49))

    # Open JSON manifest file
    manifest_path = manifest_path
    with open(manifest_path) as f:
        manifest = json.load(f)
    log.info(f"found {len(manifest)} scans in the manifest")

    for idx, entry in enumerate(manifest): 
        label_path = Path(entry["label_path"])
        embedding_path = Path(entry["embedding_path"])

        embedding = np.load(embedding_path)
        label = np.load(label_path)
        
        # Sanity check for embedding shape 
        if embedding.shape != EMBEDDING_SHAPE:
            log.warning(
            f"Unexpected shape {embedding.shape}"
            f"(expected {EMBEDDING_SHAPE}), skipping"
            )
            continue 
        
        # Sanity check for label shape 
        if label.shape != LABEL_SHAPE:
            log.warning(
            f"Unexpected shape {label.shape}"
            f"(expected {LABEL_SHAPE}), skipping"
            )
            continue 

        X[idx, :] = embedding
        y[idx, :] = label 
    
    log.info(f"X shape: {X.shape}  y shape: {y.shape}")
    log.info(f"Positive rate per class (min/mean/max): "
             f"{y.mean(axis=0).min():.3f} / "
             f"{y.mean(axis=0).mean():.3f} / "
             f"{y.mean(axis=0).max():.3f}")
    
    return X, y

# ──────────────────────────────────────────────
# Step 2 — Build classifier
# ──────────────────────────────────────────────

def build_classifier() -> MultiOutputClassifier: 
    """
    One logistic regression for each class, with balanced class weights
    to handle label imbalance
    """

    base = LogisticRegression(
        C=1.0,
        class_weight="balanced", # upweight positive samples automatically
        max_iter= 1000,
        solver = 'lbfgs',
        random_state=54,
    )
    return MultiOutputClassifier(base, n_jobs=-1)  # n_jobs=-1 = use all CPU cores

# ──────────────────────────────────────────────
# Step 3 — Per-fold evaluation helper
# ──────────────────────────────────────────────

def compute_fold_auroc(
        y_true:np.ndarray,
        y_prob:np.ndarray,
) -> dict : 
     """
    Compute per-class and macro AUROC for one fold.
 
    y_true : (n_val, 50) binary ground truth
    y_prob : (n_val, 50) predicted probabilities for the positive class
 
    Returns a dict with:
        per_class : list of 50 floats (NaN if class absent in val fold)
        macro     : float (mean over non-NaN classes)
    """