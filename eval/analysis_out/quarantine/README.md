# Quarantined runs — DO NOT USE THESE NUMBERS

The files in this directory are **failed runs, not results**. They are kept only
as evidence of the failure mode; nothing here may be aggregated, plotted, or
cited.

## What happened

`bilingual_v3_en_or_gpt-5.5.json` and `bilingual_v3_en_or_gpt-5.2.json` each
contain 90 records with an **empty `raw` field for all 90** and a summary of
`acc = 0.0000`.

That accuracy is an artifact, not model behavior. Both frontier endpoints are
geoblocked on direct egress (a clean HTTP 403, "not available in your region"),
so every call has to traverse a single SOCKS tunnel on `127.0.0.1:1080`. At the
time of these runs, four arms were sharing that tunnel with ~24 concurrent
long-lived requests. The tunnel began dropping connections at roughly 25s into
each request. The runner recorded each dropped call as an unparsable reply,
which its scoring logic counts as **wrong** — so a total transport failure was
written to disk as a well-formed result file with a plausible-looking accuracy
field.

Proof it was the tunnel and not the models: the identical request that failed
three consecutive times under contention succeeded in 9.3s once the other arms
were stopped and the tunnel was quiet.

## Why this is dangerous

"English cross-voice transport collapses to 0%" would have read as a headline
finding. Nothing in the file's shape distinguishes it from a real result — the
record count is right, the schema is right, and the accuracy field is populated.
Only auditing `raw` for emptiness reveals it.

## The rule this establishes

1. Run **one arm at a time** through the tunnel. Never let concurrent request
   count approach the tunnel's limit (~8–10 long requests).
2. **Audit every output file for empty replies before any number is used.** An
   empty `raw` means the call failed; it does not mean the model answered wrong.
3. A parse-failure-counts-as-wrong policy is correct for a genuine bad reply and
   dangerous for a dropped connection. The two must be distinguished at the
   audit step, because the scoring step cannot tell them apart.

`run_remaining_arms.sh` enforces (1) and (2).

## Status

These two models were re-run serially. The valid results live at
`eval/analysis_out/bilingual_v3_en_or_gpt-5.5.json` and
`..._or_gpt-5.2.json`. The published `results_v2/`, `catlab/results/`, and
`eval/results/` sets were audited and contain **zero** empty replies.
