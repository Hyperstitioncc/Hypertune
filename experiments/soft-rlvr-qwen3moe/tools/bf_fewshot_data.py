"""Prompt-only baseline for Brainfuck: the test problems of DATA with a description of the language and K
solved train problems placed before every prompt (no adapter). Scored with CKPT=none.
    python tools/bf_fewshot_data.py DATA OUT K [SEED]     ->  OUT/test.jsonl, OUT/meta.json
K = 0: one solved train problem of every task family (the prompt then shows every kind of task the test set has)."""
import json, os, random, sys

data, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
SPEC = ("Brainfuck is a programming language with eight commands. They act on a tape of 30,000 byte cells, all "
        "zero at the start, and a pointer to one cell:\n"
        ">  move the pointer one cell right\n<  move the pointer one cell left\n"
        "+  add 1 to the current cell (255 wraps to 0)\n-  subtract 1 from the current cell (0 wraps to 255)\n"
        ".  output the current cell as a character\n"
        ",  read one input character into the current cell (0 when the input has ended)\n"
        "[  if the current cell is 0, jump to after the matching ]\n"
        "]  if the current cell is not 0, jump back to after the matching [\n"
        "Every other character is ignored.\n\nSolved problems:\n\n")
load = lambda s: [json.loads(l) for l in open(f"{data}/{s}.jsonl")]
train = load("train")
shots = random.Random(seed).sample(train, k) if k else list({x["family"]: x for x in random.Random(seed).sample(train, len(train))}.values())
head = SPEC + "".join(x["prompt"] + x["target"] + "\n\n" for x in shots)
os.makedirs(out, exist_ok=True)
rows = [{**x, "prompt": head + x["prompt"]} for x in load("test")]
with open(f"{out}/test.jsonl", "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
json.dump({"domain": "bf", "source": f"{data} test with a language description and {len(shots)} solved train problems in the prompt",
           "seed": seed, "shots": [x["id"] for x in shots], "head_chars": len(head)}, open(f"{out}/meta.json", "w"), indent=1)
print(f"{out}: {len(rows)} test prompts, {len(shots)} shots, {len(head)} characters before each problem; families "
      f"{sorted({x['family'] for x in shots})}")
