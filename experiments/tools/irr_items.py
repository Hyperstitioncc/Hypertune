"""Off-task prompts for the write penalty, in the shared library's --irr-items format (a JSON list of
{"prompt", "response"}): the train rows of every DATA_DIR (tools/build_neg_pool.py sources, filler tasks).
    python tools/irr_items.py OUT.json DATA_DIR [DATA_DIR ...]"""
import json, sys

out, dirs = sys.argv[1], sys.argv[2:]
items = []
for d in dirs:
    with open(f"{d}/train.jsonl") as f:
        items += [{"prompt": r["prompt"], "response": r["target"]} for r in map(json.loads, f) if r["target"]]
json.dump(items, open(out, "w"))
print(f"{out}: {len(items)} prompts from {len(dirs)} sources")
