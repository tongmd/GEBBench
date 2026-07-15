# GEB-Bench evaluation

```bash
python3 eval/build_items.py                 # eval/items.json from library + stories + scenes
python3 eval/run_eval.py --base URL --model NAME [--vision] [--tasks a,b] [--tag TAG]
python3 eval/analyze.py                     # paper/tables/*.tex + eval/results/_aggregate.json
```

Open models are served with vLLM (OpenAI-compatible); text-only models run
`story2motif,theorem2motif`. All items are k-way multiple choice, judge-free.
