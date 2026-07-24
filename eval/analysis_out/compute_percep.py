#!/usr/bin/env python3
"""Compute CLIP-base32 and SigLIP image embeddings for the 125 GEB renders.

Reuses the canonical 'paths' ordering from exp1_geometry/embeddings.npz so the
new perceptual matrices align row-for-row with the existing DINOv2 geometry.
Images are read from /tmp/geb_renders/<motif>/<name>.png (rsynced from kiti).
Writes eval/analysis_out/percep_clip.npz and percep_siglip.npz, each with keys
'paths' (N,) and 'emb' (N, D).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
from PIL import Image

ROOT = Path("/Users/stone/GEB-benchmark")
EXP = ROOT / "exp1_geometry"
IMG_ROOT = Path("/tmp/geb_renders")
OUT = ROOT / "eval" / "analysis_out"
OUT.mkdir(parents=True, exist_ok=True)

DEV = "cpu"
torch.set_num_threads(4)


def load_paths():
    z = np.load(EXP / "embeddings.npz", allow_pickle=True)
    return [str(p) for p in z["paths"]]


def open_images(paths):
    ims = []
    for p in paths:
        fp = IMG_ROOT / p
        assert fp.exists(), f"missing render: {fp}"
        ims.append(Image.open(fp).convert("RGB"))
    return ims


def compute_clip(paths, ims):
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    proc = CLIPProcessor.from_pretrained(name)
    model = CLIPModel.from_pretrained(name).to(DEV).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(ims), 8):
            batch = ims[i:i + 8]
            inp = proc(images=batch, return_tensors="pt").to(DEV)
            v = model.vision_model(pixel_values=inp["pixel_values"])
            feat = model.visual_projection(v.pooler_output)
            embs.append(feat.cpu().numpy())
    return np.concatenate(embs).astype(np.float32)


def compute_siglip(paths, ims):
    from transformers import AutoImageProcessor, SiglipVisionModel
    name = "google/siglip-base-patch16-224"
    proc = AutoImageProcessor.from_pretrained(name)
    model = SiglipVisionModel.from_pretrained(name).to(DEV).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(ims), 8):
            batch = ims[i:i + 8]
            inp = proc(images=batch, return_tensors="pt").to(DEV)
            v = model(pixel_values=inp["pixel_values"])
            feat = v.pooler_output  # SigLIP: pooled head output = image feats
            embs.append(feat.cpu().numpy())
    return np.concatenate(embs).astype(np.float32)


def main():
    paths = load_paths()
    ims = open_images(paths)
    print(f"loaded {len(ims)} renders", flush=True)

    clip = compute_clip(paths, ims)
    np.savez(OUT / "percep_clip.npz", paths=np.array(paths), emb=clip)
    print(f"CLIP-base32 -> {clip.shape}  wrote percep_clip.npz", flush=True)

    siglip = compute_siglip(paths, ims)
    np.savez(OUT / "percep_siglip.npz", paths=np.array(paths), emb=siglip)
    print(f"SigLIP -> {siglip.shape}  wrote percep_siglip.npz", flush=True)


if __name__ == "__main__":
    main()
