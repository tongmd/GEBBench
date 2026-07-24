#!/usr/bin/env python3
"""Exp1 — image embeddings (runs on kiti GPU, env ~/micromamba/envs/interp).

Reads every PNG under IMAGES_DIR (layout <motif>/<name>.png), embeds with
DINOv2-base (CLS token) and CLIP ViT-L/14 (image features), writes
embeddings.npz with keys: paths (list "motif/name.png"), dino (N,768),
clip (N,768).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGES_DIR = Path(sys.argv[1])
OUT = Path(sys.argv[2])
DEV = "cuda" if torch.cuda.is_available() else "cpu"

paths = sorted(p for p in IMAGES_DIR.glob("*/*.png"))
print(f"{len(paths)} images, device={DEV}", flush=True)

from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()

dinos, clips, keep = [], [], []
with torch.no_grad():
    for i in range(0, len(paths), 8):
        batch = paths[i:i + 8]
        ims = [Image.open(p).convert("RGB") for p in batch]
        d_in = dino_proc(images=ims, return_tensors="pt").to(DEV)
        d_out = dino(**d_in).last_hidden_state[:, 0]  # CLS
        c_in = clip_proc(images=ims, return_tensors="pt").to(DEV)
        v_out = clip.vision_model(pixel_values=c_in["pixel_values"])
        c_out = clip.visual_projection(v_out.pooler_output)
        dinos.append(d_out.cpu().numpy())
        clips.append(c_out.cpu().numpy())
        keep += [str(p.relative_to(IMAGES_DIR)) for p in batch]
        print(f"  {i + len(batch)}/{len(paths)}", flush=True)

np.savez(OUT, paths=np.array(keep),
         dino=np.concatenate(dinos), clip=np.concatenate(clips))
print("wrote", OUT, flush=True)
