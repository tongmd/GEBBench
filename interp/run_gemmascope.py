#!/usr/bin/env python3
"""GemmaScope SAE feature extraction over the GEB-Bench voice triplets.

For each of the 25 motifs we have three "voices" of the same abstract structure:
  - story    : Chinese folk story whose telling enacts the structure
               (byte-identical to eval/items.json story2motif texts)
  - theorem  : masked theorem text (byte-identical to eval theorem2motif texts,
               motif-revealing names replaced by □ — blocks lexical leakage)
  - invariant: one-line invariant (data/motif_library.json invariant_zh)

We run Gemma-2 once per text, grab the post-block residual stream at the
requested layers (HF hidden_states[L+1] == TransformerLens resid_post[L],
which is what the gemma-scope-*-pt-res SAEs were trained on), encode with the
canonical JumpReLU SAE of each layer, and save:

  results/acts_{tag}.npz
      pooled feature vectors  mean_L{L}  (75, d_sae)  float16
                              max_L{L}   (75, d_sae)  float16
      per-token sparse top-k  tok_idx_L{L}_{i}, tok_val_L{L}_{i}  (for probe)
  results/meta_{tag}.json     motif ids, voices, config
  results/quickmetrics_{tag}.json   immediate readout: cross-voice retrieval
                                    + diag-vs-offdiag gap per layer

Usage (GPU box):
  python run_gemmascope.py --layers 4,8,12,16,20,24            # full zh run
  python run_gemmascope.py --smoke                             # 2 motifs, 1 layer
  python run_gemmascope.py --lang en --layers 12,20            # EN-story robustness
  python run_gemmascope.py --dry-run                           # data check, no torch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = Path(__file__).resolve().parent / "results"

VOICES = ("story", "theorem", "invariant")


# ---------------------------------------------------------------- data

def load_texts(lang: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Return (motif_ids, {motif_id: {voice: text}})."""
    lib = json.loads((ROOT / "data" / "motif_library.json").read_text())
    items = json.loads((ROOT / "eval" / "items.json").read_text())["items"]
    story = {i["answer"]: i["text"] for i in items if i["task"] == "story2motif"}
    thm = {i["answer"]: i["text"] for i in items if i["task"] == "theorem2motif"}
    if lang == "en":
        sv = json.loads((ROOT / "data" / "stories_v2.json").read_text())
        story = {s["motif_id"]: s["story_en"] for s in sv}
    ids = [m["id"] for m in lib]
    texts = {}
    for m in lib:
        mid = m["id"]
        assert mid in story and mid in thm, f"missing voice for {mid}"
        texts[mid] = {
            "story": story[mid],
            "theorem": thm[mid],
            "invariant": m["invariant_zh"],
        }
    return ids, texts


# ---------------------------------------------------------------- model / SAE

def load_sae(release: str, sae_id: str, device: str):
    from sae_lens import SAE
    out = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae = out[0] if isinstance(out, tuple) else out  # API changed across versions
    return sae.to(device)


def get_residuals(model, tokenizer, text: str, layers: list[int], device: str,
                  max_tokens: int):
    """One forward pass; return {layer: (seq, d_model) float32 on device},
    with the BOS position dropped."""
    import torch
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: [emb, layer0_out, ..., layerN-1_out]
    return {L: hs[L + 1][0, 1:, :].float() for L in layers}  # drop BOS


# ---------------------------------------------------------------- quick metrics

def quick_metrics(mean_vecs, ids):
    """mean_vecs: {layer: (n_motifs, 3, d_sae)} numpy. Returns per-layer
    cross-voice (story vs theorem) diag-offdiag gap + retrieval acc."""
    import numpy as np
    res = {}
    for L, V in mean_vecs.items():
        s, t = V[:, 0, :], V[:, 1, :]  # story, theorem
        sn = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-8)
        tn = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
        sim = sn @ tn.T  # (n, n)
        n = sim.shape[0]
        diag = float(np.mean(np.diag(sim)))
        off = float((sim.sum() - np.trace(sim)) / (n * n - n))
        top1 = float(np.mean(sim.argmax(axis=1) == np.arange(n)))
        res[str(L)] = {"diag_cos": diag, "offdiag_cos": off,
                       "gap": diag - off, "retrieval_top1": top1,
                       "chance": 1.0 / n}
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical",
                    help="use gemma-scope-9b-pt-res-canonical with --model google/gemma-2-9b")
    ap.add_argument("--width", default="16k")
    ap.add_argument("--layers", default="4,8,12,16,20,24",
                    help="comma-separated residual layers (0-indexed blocks)")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh",
                    help="zh = exact eval texts for all 3 voices (primary); "
                         "en = story_en robustness (theorem/invariant stay zh)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tok-topk", type=int, default=30,
                    help="per-token top-k features kept for the probe")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--smoke", action="store_true", help="2 motifs, first layer only")
    ap.add_argument("--dry-run", action="store_true",
                    help="load + print data, no torch (works on the Mac)")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    ids, texts = load_texts(args.lang)
    if args.smoke:
        ids = ids[:2]
        layers = layers[:1]
    tag = f"{args.lang}" + ("_smoke" if args.smoke else "")

    print(f"[data] {len(ids)} motifs x {len(VOICES)} voices, lang={args.lang}")
    for mid in ids[:2]:
        for v in VOICES:
            print(f"  {mid}/{v}: {texts[mid][v][:60]!r}... ({len(texts[mid][v])} chars)")
    if args.dry_run:
        print("[dry-run] data OK, exiting before torch.")
        return

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    print(f"[model] loading {args.model} ({args.dtype}) on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
        attn_implementation="eager")  # gemma-2 recommends eager
    model.eval()
    n_layers = model.config.num_hidden_layers
    assert all(0 <= L < n_layers for L in layers), \
        f"layers must be in [0,{n_layers}) for {args.model}"

    saes = {}
    for L in layers:
        sae_id = f"layer_{L}/width_{args.width}/canonical"
        print(f"[sae] loading {args.sae_release} :: {sae_id}")
        saes[L] = load_sae(args.sae_release, sae_id, args.device)
    d_sae = saes[layers[0]].cfg.d_sae
    print(f"[sae] d_sae={d_sae}")

    n = len(ids)
    mean_pool = {L: np.zeros((n, len(VOICES), d_sae), dtype=np.float32) for L in layers}
    max_pool = {L: np.zeros((n, len(VOICES), d_sae), dtype=np.float32) for L in layers}
    tok_sparse = {L: {} for L in layers}  # (i, voice) -> (idx, val) arrays

    for i, mid in enumerate(ids):
        for vi, voice in enumerate(VOICES):
            text = texts[mid][voice]
            resid = get_residuals(model, tokenizer, text, layers,
                                  args.device, args.max_tokens)
            for L in layers:
                with torch.no_grad():
                    acts = saes[L].encode(resid[L])  # (seq, d_sae), JumpReLU >= 0
                a = acts.float()
                mean_pool[L][i, vi] = a.mean(0).cpu().numpy()
                max_pool[L][i, vi] = a.amax(0).cpu().numpy()
                k = min(args.tok_topk, d_sae)
                val, idx = torch.topk(a, k, dim=1)
                tok_sparse[L][(i, vi)] = (
                    idx.cpu().numpy().astype(np.int32),
                    val.cpu().numpy().astype(np.float16))
        print(f"[run] {i + 1}/{n} {mid} done")

    # ---- save
    arrays = {}
    for L in layers:
        arrays[f"mean_L{L}"] = mean_pool[L].astype(np.float16)
        arrays[f"max_L{L}"] = max_pool[L].astype(np.float16)
        for (i, vi), (idx, val) in tok_sparse[L].items():
            arrays[f"tokidx_L{L}_{i}_{vi}"] = idx
            arrays[f"tokval_L{L}_{i}_{vi}"] = val
    acts_path = out_dir / f"acts_{tag}.npz"
    np.savez_compressed(acts_path, **arrays)

    meta = {"model": args.model, "sae_release": args.sae_release,
            "width": args.width, "layers": layers, "lang": args.lang,
            "voices": list(VOICES), "motif_ids": ids, "d_sae": int(d_sae),
            "tok_topk": args.tok_topk, "max_tokens": args.max_tokens}
    (out_dir / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))

    qm = quick_metrics(mean_pool, ids)
    (out_dir / f"quickmetrics_{tag}.json").write_text(json.dumps(qm, indent=2))
    print(f"[save] {acts_path}")
    print("[quick] story<->theorem cross-voice (mean-pool cosine):")
    for L, m in qm.items():
        print(f"  layer {L}: diag={m['diag_cos']:.3f} offdiag={m['offdiag_cos']:.3f} "
              f"gap={m['gap']:.3f} retrieval@1={m['retrieval_top1']:.2f} "
              f"(chance {m['chance']:.2f})")


if __name__ == "__main__":
    sys.exit(main())
