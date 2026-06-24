"""
TopAneu 2026 - Task 1
Baseline v0 - Step 1: Feature Extraction
 
Loads the pretrained frozen SwinUNETR encoder and extracts one 768-dim
embedding vector per scan via global average pooling.
 
Input  : preprocessed/ folder (images/ + labels/ + manifest.json)
Output : preprocessed/embeddings/ folder with one .npy per scan
 
Usage:
    python extract_features.py \
        --preprocessed_dir /path/to/preprocessed \
        --weights_path     /path/to/ssl_pretrained_weights.pth
"""

import argparse
import json
import logging 
from pathlib import Path
import os

import nibabel as nib 
import numpy as np 
import torch 
import torch.nn.functional as F 
from monai.apps import download_url
from monai.networks.nets.swin_unetr import SwinUNETR, filter_swinunetr
from monai.networks.utils import copy_model_state 
import time

start_time = time.time()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration — must match the pretrained weights exactly
# ──────────────────────────────────────────────

# Input shape after preprocessing (H, D, W) - no channel dim here 
IMG_SIZE = (176, 208, 144)

# Must match the pretrained checkpoint (feature_size = 48 --> bottleneck dim = 768)
FEATURE_SIZE = 48 

# Pre-trained weights URL 
SSL_WEIGHTS_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data"
    "/releases/download/0.8.1/ssl_pretrained_weights.pth"
)

# ──────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────

def build_frozen_encoder(weights_path : Path) -> SwinUNETR: 
    """
    Instantiate SwinUNETR, load pretrained encoder weights, freeze all parameter.
    Returns the model in eval mode on CPU 
    """
    # Instantiate the full SWINUNETR (we only use the encoder)
    # out_channels = 14 matches the pretrained checkpoint convention - irrelevant
    # for us since we discard the decoder, but must match to load weights cleanly 

    model = SwinUNETR(
        in_channels = 1,
        out_channels = 14,
        feature_size = FEATURE_SIZE,
        use_checkpoint=False # gradient checkpointing off (not training)
        )

    # Load the pretrained SSL weights 
    log.info(f"Loading the pretrained weights from {weights_path}")
    ssl_weights = torch.load(str(weights_path), map_location = "cpu")["model"]

    # copy_model_state + filter_swinunetr handles the key remapping between
    # the SSL pretraining checkpoint format and the SwinUNETR state_dict format
    # (e.g. "encoder.layers1..." --> "swinViT.layers1...")
    dst_dict, loaded, not_loaded = copy_model_state(model, ssl_weights, filter_func=filter_swinunetr
    )

    model.load_state_dict(dst_dict, strict=False)
    log.info(f"Weights loaded : {len(loaded)} layers loaded, {len(not_loaded)} not loaded")

    # Freeze all parameters - no gradients will flow through the encoder
    for param in model.parameters():
        param.requires_grad = False 
    
    model.eval()
    log.info("Encoder frozen and set to eval mode")
    return model

# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────

def extract_embedding(model: SwinUNETR, img_array: np.ndarray) -> np.ndarray:
    """
    Extract a single 768-dim embedding from a preprocessed scan.
    
    img_array : numpy array of shape (H, W, D), already normalized
    returns : numpy array of shape (768,)
    """
    # Add batch and channel dims : (H, W, D) -> (1, 1, H, W, D)
    tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad(): 
        # swinViT is the Swin Transformer backbone inside SwinUNETR
        # It returns a list of feature maps at each stage. 
        # We take the last one - the bottleneck - which has the richest
        # asbtract features and the smallest spatial dimensions. 
        hidden_states = model.swinViT(tensor, normalize=True)
        bottleneck = hidden_states[-1] # shape : (1, 768, H', W', D')
    
    # Global average pooling : collapse spatial dims -> (1, 768, 1, 1)
    pooled = F.adaptive_avg_pool3d(bottleneck, output_size=(1, 1, 1))

    # Flaten to 1D embedding vector of shape (768, )
    embedding = pooled.squeeze().numpy()
    return embedding

# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def extract_all_features(preprocessed_dir:Path, weights_path:Path) -> None: 

    # Download weights if not already present 
    #if not weights_path.exists():
        #log.info(f"Weights not found at {weights_path}, downloading...")
        #download_url(SSL_WEIGHTS_URL, str(weights_path))
    
    # Load manifest
    manifest_path = "data/images_preprocessed/manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    log.info(f"found {len(manifest)} scans in the manifest")

    # Build frozen encoder 
    model = build_frozen_encoder(weights_path)

    # Prepare output directory 
    embeddings_dir = preprocessed_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    stats = {"ok" : 0, "errors" : 0}

    for entry in manifest: 
        case_id = entry["case_id"]
        img_path = Path(entry["image_path"])
        modality = entry["modality"]

        log.info(f"Extracting [{modality}], {case_id}")

        # Load preprocessed image 
        try: 
            img_array = nib.load(str(img_path)).get_fdata()
        except Exception as e:
            log.error(f"failed to load {img_path} : {e}")
            stats["errors"] += 1
            continue 
    
        # Sanity check shape
        if img_array.shape != IMG_SIZE:
            log.warning(
                f"Unexpected shape {img_array.shape} for {case_id}"
                f"(expected {IMG_SIZE}), skipping"
            )
            stats["errors"] += 1
            continue 

        # Extract embedding
        try:
            embedding = extract_embedding(model, img_array)
        except Exception as e:
            log.error(f"Feature extraction failed for {case_id}: {e}")
            stats["errors"] += 1
            continue 

        # Save embedding 
        output_path = embeddings_dir / f"{case_id}_embedding.npy"
        np.save(str(output_path), embedding)

        log.info(f" -> embedding shape : {embedding.shape}")
        stats["ok"] += 1

 # Update manifest to include embedding paths
    for entry in manifest:
        case_id = entry["case_id"]
        emb_path = embeddings_dir / f"{case_id}_embedding.npy"
        entry["embedding_path"] = str(emb_path)
 
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
 
    log.info("=" * 50)
    log.info(f"Done. Extracted: {stats['ok']}  Errors: {stats['errors']}")
    log.info(f"Embeddings saved to {embeddings_dir}")
    log.info(f"Manifest updated at {manifest_path}")

extract_all_features(Path("data/images_preprocessed/images"), Path("data/images_preprocessed/ssl_pretrained_weights.pth"))

end_time = time.time()
execution_time = end_time - start_time
print(f"Pipeline executed in {execution_time:.2f} seconds")