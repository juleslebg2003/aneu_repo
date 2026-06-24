"""
TopAneu 2026 - Task 1
Step 2: Preprocessing & Normalization Pipeline
 
Input  : raw topaneu/ folder (images/ + location_masks/)
Output : preprocessed/ folder with normalized .nii.gz files + binary label vectors
 
Usage:
    python preprocess.py --data_dir /path/to/topaneu --output_dir /path/to/preprocessed
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from monai.data import Dataset
import logging
import time
import json

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CropForegroundd,
    SpatialPadd,
    CenterSpatialCropd,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

start_time = time.time()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# Target voxel spacing (in mm) - shared across CTA and MRA
# 1mm isotropic spacing is a safe default for CPU-only envrironments (~23 MB per scan)
TARGET_SPACING = (1.0, 1.0, 1.0)

# Setting the target shape 
TARGET_SHAPE = (176, 208, 144)

#CTA intensity window (HU). Covers vessl lumen well. 
CTA_WINDOW_MIN = -100
CTA_WINDOW_MAX = 800

#MRA : z-score normalization (no fixed window - intensity is arbitrary)
MRA_NONZERO_ONLY = True  # Only compute mean/std on non-zero voxels (background is zero)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def infer_modality (filename : str) -> str:
    """
    Infer the modality (CTA or MRA) from the filename.
    Assumes that filenames contain either 'ct' or 'mr'.
    """
    filename_lower = filename.lower()
    if "ct" in filename_lower:
        return "CTA"
    if "mr" in filename_lower:
        return "MRA"
    raise ValueError(f"Cannot infer modality from filename: {filename}")


def load_label_map(json_path):
    "Load JSON file"
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)["labels"]
    return data

label_map = load_label_map('/Users/julesperbet/aneu_repo/data/location_mapping.json') 

def load_location_labels(mask_path : Path, n_classes : int = 50) -> np.ndarray:
    """
    Build a binary multi-label vector of shape (n_classes,) directly from
    the location mask. Each unique non-zero integer value in the mask
    corresponds to an aneurysm location class (1-indexed).
    """
    mask = nib.load(mask_path).get_fdata().astype(int)  # Ensure the mask is of integer type
    vector = np.zeros(n_classes - 1, dtype=np.float32)  # Exclude background
    for v in np.unique(mask):
        if v > 0 and v < n_classes:  # Ensure we don't go out of bounds
            vector[v - 1] = 1

    return vector   

def build_cta_transforms(target_spacing : tuple) -> Compose : 
    """
    CTA pipeline : 
        1. Load the image
        2. Ensure channel 
        3. Resample to target spacing
        4. Window + rescale to [0, 1]
        
    Note : no reorientation needed - all scans are already in LPS+.
    Axis 0 = Left→Right. Remember this when implementing L/R flip augmentation.
    
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),  # Ensure channel first
        Spacingd(
            keys=["image"], 
            pixdim=target_spacing, 
            mode="bilinear"
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image", 
        ), 
        SpatialPadd(
            keys = ["image"],
            spatial_size = TARGET_SHAPE,
        ),
        CenterSpatialCropd(
            keys = ["image"], 
            roi_size = TARGET_SHAPE,
        ),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=CTA_WINDOW_MIN, 
            a_max=CTA_WINDOW_MAX, 
            b_min = 0.0,
            b_max = 1.0,
            clip=True
        )
    ])

def build_mra_transforms(target_spacing : tuple) -> Compose :
    """
    MRA pipeline :
        1. Load the image 
        2. Ensure channel first
        3. Resample to target spacing
        4. Z-score normalization (non-zero voxels only)
        
    Note : no reorientation needed - all scans are already in LPS+.
    Axis 0 = Left→Right. Remember this when implementing L/R flip augmentation.
    
    """
    return Compose([
        LoadImaged(keys = ["image"]),
        EnsureChannelFirstd(keys=["image"]),  # Ensure channel first
        Spacingd(
            keys=["image"], 
            pixdim=target_spacing, 
            mode="bilinear"
        ),
         CropForegroundd(
            keys=["image"],
            source_key="image",   
        ), 
        SpatialPadd(
            keys = ["image"],
            spatial_size = TARGET_SHAPE,
        ),
        CenterSpatialCropd(
            keys = ["image"], 
            roi_size = TARGET_SHAPE,
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=MRA_NONZERO_ONLY,
            channel_wise=True
        )
    ])

# ──────────────────────────────────────────────
# Main preprocessing loop
# ──────────────────────────────────────────────

def preprocess_dataset(images_dir : Path, masks_dir : Path, output_dir : Path) -> None :

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    # Prepare output directories
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Build MONAI transforms for CTA and MRA
    cta_transforms = build_cta_transforms(TARGET_SPACING)
    mra_transforms = build_mra_transforms(TARGET_SPACING)

    # Collect all image files
    image_paths = sorted (images_dir.glob("*.nii.gz"))
    log.info(f"Found {len(image_paths)} images in {images_dir} scans to process.")

    stats = {"CTA" : 0, "MRA" : 0, "errors" : 0}

    for img_path in image_paths:
        case_id = img_path.name.replace("_0000.nii.gz","") # e.g. topaneu_center2_mr_002
        mask_path = masks_dir / f"{case_id}.nii.gz"

        if not mask_path.exists() :
            log.warning(f"No location mask for {case_id}, skipping")
            stats["errors"] += 1
            continue
    
        try :
            modality = infer_modality(img_path.name)
        except ValueError as e : 
            log.error(e)
            stats["errors"] += 1
            continue
    
        log.info(f"processing [{modality}] {case_id}")

        # ── Apply modality-specific MONAI transforms ────────────────────────
        sample = {"image" : str(img_path)}
        transforms = cta_transforms if modality == "CTA" else mra_transforms
        
        #try: 
        processed = transforms(sample)
        #except Exception as e :
            #log.error(f"Transform failed for {case_id}: {e}")
            #stats["errors"] += 1
            #continue 
        
        processed_array = processed["image"].numpy() # (1,H, W, D)

        assert not np.isnan(processed_array).any(), f"NaN in {case_id}"
        assert not np.isinf(processed_array).any(), f"Inf in {case_id}"

        if modality == "CTA":
                assert processed_array.min() >= 0.0 and processed_array.max() <= 1.0, \
                    f"CTA range out of [0,1] for {case_id}"
        
        # ── Save preprocessed image ─────────────────────────────────────────   
        out_img_path = output_images_dir / f"{case_id}.nii.gz"
        affine = processed["image"].meta["affine"].numpy()
        new_img = nib.Nifti1Image(processed_array[0], affine=affine)
        nib.save(new_img, str(out_img_path))

        # ── Build and save label vector from location mask ──────────────────
        label_vec = load_location_labels(mask_path)
        out_label_path = output_labels_dir / f"{case_id}_labels.npy"
        np.save(str(out_label_path), label_vec)
        
        log.info(
            f" -> shape {processed_array.shape[1:]}"
            f" positive classes : {int(label_vec.sum())}"
        )
        stats[modality] += 1

    # Final summary 
    log.info("=" * 50)
    log.info(f"Done. CTA : {stats['CTA']} MRA : {stats['MRA']} Errors : {stats['errors']}")

    manifest = []
    for img_path in sorted(output_images_dir.glob("*.nii.gz")):
        case_id = img_path.stem.replace(".nii", "")
        label_path = output_labels_dir / f"{case_id}_labels.npy"
        modality = infer_modality(img_path.name)
        manifest.append({
            "case_id":    case_id,
            "modality":   modality,
            "image_path": str(img_path),
            "label_path": str(label_path),
        })
 
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest saved → {manifest_path}")

preprocess_dataset(Path("/Users/julesperbet/aneu_repo/data/images"), Path("/Users/julesperbet/aneu_repo/data/location_masks"), Path("/Users/julesperbet/aneu_repo/data/images_preprocessed"))

end_time = time.time()
execution_time = end_time - start_time
print(f"Pipeline executed in {execution_time:.2f} seconds")