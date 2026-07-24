#!/usr/bin/env python3
"""Bilingual structural-transport analysis: story2motif ZH vs EN.

Inputs:
  eval/results/api_<tag>.json          (shipped ZH run, story2motif subset)
  eval/analysis_out/story_en_api_<tag>.json  (EN replication)

Outputs:
  eval/analysis_out/bilingual.json
  eval/analysis_out/bilingual_transport.pdf/.png

Stats:
  (a) per-model EN vs ZH accuracy, exact McNemar on the 25 paired items;
  (b) transport consistency P(correct_EN|correct_ZH) and converse;
      error identity when both wrong (same predicted wrong motif?);
  (c) does the ZH->EN gap differ across models? (heterogeneity of the
      per-item language effect across models via exact tests on discordant
      counts + a pooled logistic view)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "analysis_out"

TAGS = ["api_gpt-5.5", "api_gpt-5.2", "api_claude-opus-4-8"]
PRETTY = {"api_gpt-5.5": "GPT-5.5", "api_gpt-5.2": "GPT-5.2",
          "api_claude-opus-4-8": "Claude Opus 4.8"}


def load_pair(tag):
    zh = json.loads((ROOT / "eval" / "results" / f"{tag}.json").read_text())
    en = json.loads((OUT / f"story_en_{tag}.json").read_text())
    zh_recs = {r["id"].split(":")[1]: r for r in zh["records"]
               if r["task"] == "story2motif"}
    en_recs = {r["id"].split(":")[1]: r for r in en["records"]}
    assert set(zh_recs) == set(en_recs), f"{tag}: item mismatch"
    ids = sorted(zh_recs)
    return ids, zh_recs, en_recs


def mcnemar_exact(b, c):
    """Exact two-sided McNemar: b = ZH-only correct, c = EN-only correct."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n, 0.5).pvalue * 1)  # two-sided default


def main():
    results = {}
    per_item_matrix = {}
    for tag in TAGS:
        ids, zh, en = load_pair(tag)
        n = len(ids)
        zc = np.array([zh[i]["correct"] for i in ids])
        ec = np.array([en[i]["correct"] for i in ids])
        both = int((zc & ec).sum())
        zh_only = int((zc & ~ec).sum())
        en_only = int((~zc & ec).sum())
        neither = int((~zc & ~ec).sum())
        p_mcn = float(stats.binomtest(min(zh_only, en_only),
                                      zh_only + en_only, 0.5).pvalue) \
            if zh_only + en_only > 0 else 1.0
        # transport consistency
        p_en_given_zh = both / max(1, both + zh_only)
        p_zh_given_en = both / max(1, both + en_only)
        # error identity among items wrong in BOTH languages
        both_wrong = [i for i in ids if not zh[i]["correct"] and not en[i]["correct"]]
        same_wrong = [i for i in both_wrong
                      if zh[i]["pred"] is not None and zh[i]["pred"] == en[i]["pred"]]
        err_detail = [{"item": i, "answer": zh[i]["answer"],
                       "pred_zh": zh[i]["pred"], "pred_en": en[i]["pred"],
                       "same_wrong_motif": zh[i]["pred"] == en[i]["pred"]
                       and zh[i]["pred"] is not None}
                      for i in both_wrong]
        # chance rate of same wrong pick if errors were independent uniform over 24 wrong options
        results[tag] = {
            "model": PRETTY[tag], "n_items": n,
            "acc_zh": round(float(zc.mean()), 4),
            "acc_en": round(float(ec.mean()), 4),
            "gap_zh_minus_en": round(float(zc.mean() - ec.mean()), 4),
            "contingency": {"both_correct": both, "zh_only": zh_only,
                            "en_only": en_only, "neither": neither},
            "mcnemar_exact_p": round(p_mcn, 4),
            "P_correct_en_given_correct_zh": round(p_en_given_zh, 4),
            "P_correct_zh_given_correct_en": round(p_zh_given_en, 4),
            "cohen_kappa": round(float(
                (both + neither) / n - ((both + zh_only) * (both + en_only)
                 + (en_only + neither) * (zh_only + neither)) / n**2) /
                max(1e-9, (1 - ((both + zh_only) * (both + en_only)
                 + (en_only + neither) * (zh_only + neither)) / n**2)), 4),
            "both_wrong_n": len(both_wrong),
            "both_wrong_same_pred_n": len(same_wrong),
            "both_wrong_detail": err_detail,
        }
        per_item_matrix[tag] = {i: {"zh": bool(zh[i]["correct"]),
                                    "en": bool(en[i]["correct"])} for i in ids}

    # (c) does the gap differ across models?
    # discordant counts per model -> exact heterogeneity via Fisher on 3x2 table
    tab = np.array([[results[t]["contingency"]["zh_only"],
                     results[t]["contingency"]["en_only"]] for t in TAGS])
    # Freeman-Halton style via chi2 (small counts -> also report permutation)
    if tab.sum() > 0 and (tab.sum(axis=1) > 0).all():
        chi2, p_het, dof, _ = stats.chi2_contingency(tab, correction=False)
        het = {"table_zhonly_enonly": tab.tolist(),
               "chi2": round(float(chi2), 3), "dof": int(dof),
               "p_heterogeneity": round(float(p_het), 4),
               "note": "chi2 on discordant-pair direction; counts are tiny, treat as descriptive"}
    else:
        het = {"table_zhonly_enonly": tab.tolist(),
               "note": "insufficient discordant pairs for a heterogeneity test"}

    # pooled McNemar across the 3 models (treat model x item as strata)
    B = int(tab[:, 0].sum()); C = int(tab[:, 1].sum())
    pooled_p = float(stats.binomtest(min(B, C), B + C, 0.5).pvalue) if B + C else 1.0

    # cross-model shared-error structure on items wrong in both languages
    shared = {}
    for i in sorted(per_item_matrix[TAGS[0]]):
        row = {t: per_item_matrix[t][i] for t in TAGS}
        n_zh_wrong = sum(not row[t]["zh"] for t in TAGS)
        n_en_wrong = sum(not row[t]["en"] for t in TAGS)
        if n_zh_wrong + n_en_wrong > 0:
            shared[i] = {"zh_wrong_models": n_zh_wrong, "en_wrong_models": n_en_wrong}

    out = {
        "task": "story2motif bilingual transport (ZH shipped vs EN replication)",
        "n_items": 25, "chance": 0.04,
        "models": results,
        "pooled": {"zh_only": B, "en_only": C,
                   "pooled_mcnemar_exact_p": round(pooled_p, 4),
                   "pooled_acc_zh": round(float(np.mean(
                       [results[t]["acc_zh"] for t in TAGS])), 4),
                   "pooled_acc_en": round(float(np.mean(
                       [results[t]["acc_en"] for t in TAGS])), 4)},
        "gap_heterogeneity_across_models": het,
        "items_with_any_error": shared,
    }
    (OUT / "bilingual.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "items_with_any_error"},
                     indent=1, default=str)[:4000])

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    x = np.arange(len(TAGS)); w = 0.36
    zh_acc = [results[t]["acc_zh"] for t in TAGS]
    en_acc = [results[t]["acc_en"] for t in TAGS]
    ax.bar(x - w/2, zh_acc, w, label="Chinese (shipped)", color="#4477aa")
    ax.bar(x + w/2, en_acc, w, label="English (replication)", color="#ee6677")
    ax.axhline(0.04, ls="--", c="gray", lw=1)
    ax.text(-0.45, 0.06, "chance 4%", fontsize=8, color="gray")
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[t] for t in TAGS], fontsize=9)
    ax.set_ylabel("story2motif accuracy"); ax.set_ylim(0, 1.05)
    for i, t in enumerate(TAGS):
        ax.text(i, max(zh_acc[i], en_acc[i]) + 0.03,
                f"p={results[t]['mcnemar_exact_p']:.2f}", ha="center", fontsize=8)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_title("(a) Accuracy by story language", fontsize=10)

    ax = axes[1]
    cats = ["both\ncorrect", "ZH only", "EN only", "neither"]
    bottoms = np.zeros(4)
    colors = {"api_gpt-5.5": "#4477aa", "api_gpt-5.2": "#66ccee",
              "api_claude-opus-4-8": "#ee6677"}
    for t in TAGS:
        c = results[t]["contingency"]
        vals = [c["both_correct"], c["zh_only"], c["en_only"], c["neither"]]
        ax.bar(cats, vals, 0.6, bottom=bottoms, label=PRETTY[t], color=colors[t])
        bottoms += np.array(vals)
    ax.set_ylabel("items (stacked over models)")
    ax.legend(fontsize=8)
    ax.set_title("(b) Paired outcomes per item", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "bilingual_transport.pdf")
    fig.savefig(OUT / "bilingual_transport.png", dpi=180)
    print("figure ->", OUT / "bilingual_transport.pdf")


if __name__ == "__main__":
    main()
