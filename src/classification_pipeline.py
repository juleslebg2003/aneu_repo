"""
TopAneu 2026 - Task 1
Baseline v0 - Step 2: Multi-label Logistic Regression Classifier
 
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
 
import argparse
import json
import logging
import warnings
from pathlib import Path
 
import joblib
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
 
# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
 
N_SPLITS    = 5       # number of CV folds
C           = 1.0     # logistic regression regularization strength
MAX_ITER    = 1000    # max iterations for solver convergence
N_CLASSES   = 49      # number of aneurysm location classes
RANDOM_SEED = 42
 
 
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
 
 
# ──────────────────────────────────────────────
# Step 2 — Build classifier
# ──────────────────────────────────────────────
 
def build_classifier() -> MultiOutputClassifier:
    """
    One logistic regression per class, with balanced class weights
    to handle label imbalance.
    """
    base = LogisticRegression(
        C=C,
        class_weight="balanced",   # upweights positive samples automatically
        max_iter=MAX_ITER,
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    return MultiOutputClassifier(base, n_jobs=-1)  # n_jobs=-1 = use all CPU cores
 
 
# ──────────────────────────────────────────────
# Step 3 — Per-fold evaluation helper
# ──────────────────────────────────────────────
 
def compute_fold_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute per-class and macro AUROC for one fold.
 
    y_true : (n_val, 50) binary ground truth
    y_prob : (n_val, 50) predicted probabilities for the positive class
 
    Returns a dict with:
        per_class : list of 50 floats (NaN if class absent in val fold)
        macro     : float (mean over non-NaN classes)
    """
    per_class = []
 
    for class_idx in range(N_CLASSES):
        y_true_c = y_true[:, class_idx]
        y_prob_c = y_prob[:, class_idx]
 
        # Skip classes with no positives in this fold — AUROC is undefined
        if y_true_c.sum() == 0:
            log.warning(f"  Class {class_idx:02d}: no positives in val fold, skipping")
            per_class.append(float("nan"))
            continue
 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = roc_auc_score(y_true_c, y_prob_c)
 
        per_class.append(float(auc))
 
    valid_aucs = [a for a in per_class if not np.isnan(a)]
    macro = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
 
    return {"per_class": per_class, "macro": macro}
 
 
# ──────────────────────────────────────────────
# Step 4 — Cross-validation loop
# ──────────────────────────────────────────────
 
def run_cross_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Run 5-fold MultilabelStratifiedKFold CV.
    Returns a dict with per-fold and aggregated results.
    """
    kfold = MultilabelStratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
 
    fold_results = []
    all_macro_aurocs = []
 
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        log.info(f"── Fold {fold_idx + 1}/{N_SPLITS} "
                 f"(train: {len(train_idx)}, val: {len(val_idx)}) ──")
 
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
 
        valid_classes = [
            i for i in range (N_CLASSES)
            if y_train[:, i].sum() >= 2 
        ]

        invalid_classes = [
            i for i in range (N_CLASSES)
            if y_train[:, i].sum() < 2 
        ]

        if invalid_classes:
            log.warning(f"  Skipping classes with < 2 positives in train fold: {invalid_classes}")

        # Train
        clf = build_classifier()
        clf.fit(X_train, y_train[:, valid_classes])
 
        # Predict probabilities
        # predict_proba returns a list of 50 arrays, each of shape (n_val, 2)
        # We extract column 1 (positive class probability) for each class
        y_prob = np.zeros((len(val_idx), N_CLASSES), dtype=np.float32)
        proba_list = clf.predict_proba(X_val)
        for out_idx, class_idx in enumerate(valid_classes):
            y_prob[:, class_idx] = proba_list[out_idx][:, 1]

        # Evaluate
        fold_metrics = compute_fold_auroc(y_val, y_prob)
        fold_metrics["fold"] = fold_idx + 1
        fold_results.append(fold_metrics)
 
        log.info(f"  Macro AUROC: {fold_metrics['macro']:.4f}")
        all_macro_aurocs.append(fold_metrics["macro"])
 
    # Aggregate across folds
    valid_macros = [m for m in all_macro_aurocs if not np.isnan(m)]
    mean_macro   = float(np.mean(valid_macros))
    std_macro    = float(np.std(valid_macros))
 
    log.info("=" * 50)
    log.info(f"CV Macro AUROC: {mean_macro:.4f} ± {std_macro:.4f}")
 
    # Per-class mean AUROC across folds
    per_class_across_folds = []
    for class_idx in range(N_CLASSES):
        class_aucs = [
            fold["per_class"][class_idx]
            for fold in fold_results
            if not np.isnan(fold["per_class"][class_idx])
        ]
        mean_auc = float(np.mean(class_aucs)) if class_aucs else float("nan")
        per_class_across_folds.append(mean_auc)
        if not np.isnan(mean_auc):
            log.info(f"  Class {class_idx:02d}: mean AUROC = {mean_auc:.4f} "
                     f"(over {len(class_aucs)} folds)")
 
    return {
        "n_splits":               N_SPLITS,
        "mean_macro_auroc":       mean_macro,
        "std_macro_auroc":        std_macro,
        "per_class_mean_auroc":   per_class_across_folds,
        "fold_results":           fold_results,
    }
 
 
# ──────────────────────────────────────────────
# Step 5 — Retrain on full dataset and save
# ──────────────────────────────────────────────
 
def retrain_and_save(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    cv_results: dict,
) -> None:
    """
    Retrain on all available data and save model + CV results to disk.
    """
    log.info("Retraining on full dataset...")
    clf = build_classifier()
    clf.fit(X, y)
 
    model_path = output_dir / "baseline_v0_model.joblib"
    joblib.dump(clf, str(model_path))
    log.info(f"Model saved → {model_path}")
 
    results_path = output_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    log.info(f"CV results saved → {results_path}")
 
 
# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
 
def main(preprocessed_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # Step 1 — Load data
    X, y = load_data(preprocessed_dir)
 
    # Step 4 — Cross-validation
    cv_results = run_cross_validation(X, y)
 
    # Step 5 — Retrain on full data and save
    retrain_and_save(X, y, output_dir, cv_results)
 
main(Path("data/images_preprocessed"), Path("data/images_classified"))