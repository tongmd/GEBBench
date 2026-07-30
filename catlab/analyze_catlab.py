#!/usr/bin/env python3
"""Analyze the functorial-abstraction pilot.

Per model:
  1. task accuracies (+95% Wilson CIs), per-edge breakdown
  2. conditional decomposition:
       P(square correct | both endpoints of the edge identified in T0)
       P(square correct | at least one endpoint missed)
     -> "recognizes the objects, moves the morphism?"
  3. recognition-only shortcut ceiling, simulated from the model's own T0
     answers: the shortcut picks the option whose motif equals the model's
     T0 prediction for the AFTER panel (falls back to random among 4).
     square_acc > ceiling  => transport signal beyond recognition
     square_acc <= ceiling => K3 territory (reduces to recognition)
  4. identity-trap pull: share of square/composition errors that chose the
     unchanged-structure option (equivariance failure mode), vs 1/3 chance
     among the wrong options.
  5. composition: FVR-style conditional -- P(composite right | both step
     squares right for the same seed's edges).
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - hw), min(1, c + hw)


def analyze(path: Path):
    d = json.loads(path.read_text())
    recs = d["records"]
    # An empty reply is a dropped connection, not a wrong answer. Every frontier
    # endpoint here is reached through one SOCKS tunnel, and a saturated tunnel
    # drops requests mid-flight; the runner then scores the unparsable reply as
    # wrong. So a pure transport failure can masquerade as a real accuracy --
    # two bilingual runs were lost to exactly that (90/90 empty, acc=0.0000).
    # See eval/analysis_out/quarantine/README.md.
    empty = sum(1 for r in recs if not str(r.get("raw") or "").strip())
    if empty:
        raise SystemExit(
            f"{path.name}: {empty}/{len(recs)} records have an empty reply. "
            "These are failed calls, not wrong answers -- re-run this model "
            "before analyzing.")
    by = {}
    for r in recs:
        by.setdefault(r["task"], []).append(r)
    out = {"model": d["model"]}

    # 1 ------------------------------------------------------------------
    out["tasks"] = {}
    for t, rs in by.items():
        k = sum(r["correct"] for r in rs)
        p, lo, hi = wilson(k, len(rs))
        out["tasks"][t] = {"n": len(rs), "acc": round(p, 4),
                           "ci95": [round(lo, 3), round(hi, 3)]}

    # endpoint lookup: (edge, seed, side) -> correct?
    eid = {(r["meta"]["edge"], r["meta"]["seed"], r["meta"]["side"]):
           (r["correct"], r["pred"]) for r in by.get("endpoint_id", [])}

    # 2+3+4 ----------------------------------------------------------------
    for sq in ("square_thm", "square_story"):
        rows = by.get(sq, [])
        cond = {"both": [0, 0], "not_both": [0, 0]}
        ceil_hits = 0
        id_pulls, wrongs = 0, 0
        per_edge = {}
        for r in rows:
            e, s = r["meta"]["edge"], r["meta"]["seed"]
            ok_base = eid.get((e, s, "base"), (False, None))[0]
            ok_img, pred_img = eid.get((e, s, "img"), (False, None))
            key = "both" if (ok_base and ok_img) else "not_both"
            cond[key][0] += int(r["correct"])
            cond[key][1] += 1
            pe = per_edge.setdefault(e, [0, 0])
            pe[0] += int(r["correct"])
            pe[1] += 1
            # shortcut: pick the option whose motif == model's T0 read of AFTER
            om = r.get("option_motifs") or {}
            hit = [L for L, mid in om.items() if mid == pred_img]
            if len(hit) == 1:
                ceil_hits += int(hit[0] == r["answer"])
            else:
                ceil_hits += 0.25
            if not r["correct"] and (r["pred"] or "") in "ABCD" and r["pred"]:
                wrongs += 1
                if r["pred"] == r["meta"].get("identity_option"):
                    id_pulls += 1
        n = len(rows)
        out[sq] = {
            "P_square_given_both_endpoints":
                round(cond["both"][0] / cond["both"][1], 4)
                if cond["both"][1] else None,
            "n_both": cond["both"][1],
            "P_square_given_missed_endpoint":
                round(cond["not_both"][0] / cond["not_both"][1], 4)
                if cond["not_both"][1] else None,
            "n_not_both": cond["not_both"][1],
            "shortcut_ceiling": round(ceil_hits / n, 4) if n else None,
            "acc_minus_ceiling":
                round(out["tasks"][sq]["acc"] - ceil_hits / n, 4) if n else None,
            "identity_pull": {"errors": wrongs, "chose_identity": id_pulls,
                              "share": round(id_pulls / wrongs, 3) if wrongs else None,
                              "chance": 0.333},
            "per_edge_acc": {e: round(v[0] / v[1], 3)
                             for e, v in sorted(per_edge.items())},
        }

    # id_vs_change split --------------------------------------------------
    ivc = by.get("id_vs_change", [])
    if ivc:
        st = [r for r in ivc if r["meta"]["kind"] == "structural"]
        nu = [r for r in ivc if r["meta"]["kind"] == "nuisance"]
        out["id_vs_change"] = {
            "structural_acc": round(sum(r["correct"] for r in st) / len(st), 4),
            "nuisance_acc": round(sum(r["correct"] for r in nu) / len(nu), 4),
            "n_each": len(st),
            # false-alarm rate: nuisance pairs called "structural change"
            "nuisance_called_change":
                round(sum(1 for r in nu if not r["correct"]) / len(nu), 4),
        }

    # composition ----------------------------------------------------------
    comp = by.get("composition", [])
    if comp:
        sq_by = {(r["meta"]["edge"], r["meta"]["seed"]): r["correct"]
                 for r in by.get("square_thm", [])}
        both_ok, both_ok_comp_ok, stop_g1 = 0, 0, 0
        for r in comp:
            s = r["meta"]["seed"]
            g1 = sq_by.get(("nesting_to_recursion", s))
            g2 = sq_by.get(("recursion_to_contraction", s))
            if g1 and g2:
                both_ok += 1
                both_ok_comp_ok += int(r["correct"])
            if not r["correct"] and r["pred"] == r["meta"].get("stop_after_g1_option"):
                stop_g1 += 1
        out["composition"] = {
            "acc": out["tasks"]["composition"]["acc"],
            "n_both_steps_right": both_ok,
            "P_composite_given_steps": round(both_ok_comp_ok / both_ok, 3)
                if both_ok else None,
            "errors_stopped_after_g1": stop_g1,
            "n_errors": sum(1 for r in comp if not r["correct"]),
        }
    return out


def main():
    files = sorted(glob.glob(str(ROOT / "results" / "or_*.json")))
    all_out = {}
    for f in files:
        r = analyze(Path(f))
        all_out[Path(f).stem] = r
    (ROOT / "analysis.json").write_text(json.dumps(all_out, indent=1))

    # compact report
    print(f"{'model':<22}{'endpt':>7}{'sq_thm':>8}{'sq_sty':>8}"
          f"{'ceil_t':>8}{'ceil_s':>8}{'ivc_st':>8}{'ivc_nu':>8}{'comp':>6}")
    for tag, r in all_out.items():
        t = r["tasks"]
        row = (f"{tag:<22}"
               f"{t.get('endpoint_id',{}).get('acc',0)*100:>6.1f}%"
               f"{t.get('square_thm',{}).get('acc',0)*100:>7.1f}%"
               f"{t.get('square_story',{}).get('acc',0)*100:>7.1f}%"
               f"{(r.get('square_thm') or {}).get('shortcut_ceiling',0)*100:>7.1f}%"
               f"{(r.get('square_story') or {}).get('shortcut_ceiling',0)*100:>7.1f}%"
               f"{(r.get('id_vs_change') or {}).get('structural_acc',0)*100:>7.1f}%"
               f"{(r.get('id_vs_change') or {}).get('nuisance_acc',0)*100:>7.1f}%"
               f"{t.get('composition',{}).get('acc',0)*100:>5.0f}%")
        print(row)
    print("\nconditional + identity pull:")
    for tag, r in all_out.items():
        for sq in ("square_thm", "square_story"):
            q = r.get(sq)
            if not q:
                continue
            print(f"  {tag:<20} {sq:<13} "
                  f"P(sq|endpoints)={q['P_square_given_both_endpoints']} "
                  f"(n={q['n_both']})  "
                  f"P(sq|missed)={q['P_square_given_missed_endpoint']} "
                  f"(n={q['n_not_both']})  "
                  f"id_pull={q['identity_pull']['share']} "
                  f"acc-ceil={q['acc_minus_ceiling']}")
    print("\ncomposition:")
    for tag, r in all_out.items():
        c = r.get("composition")
        if c:
            print(f"  {tag:<20} acc={c['acc']} "
                  f"P(comp|steps)={c['P_composite_given_steps']} "
                  f"stopped_after_g1={c['errors_stopped_after_g1']}/{c['n_errors']}")


if __name__ == "__main__":
    main()
