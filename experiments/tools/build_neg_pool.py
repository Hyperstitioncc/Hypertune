"""
A broad pool of "not mine" prompts for the keep-quiet penalty: many kinds of task and text, long and
short, none of them from the six test domains or the four held-out filler tasks. One data dir per
source (so a run can list the sources it may use):  $CART_ROOT/data/neg/<source>/train.jsonl
    python tools/build_neg_pool.py [N_PER_SOURCE=300] [OUT_DIR]      (OUT_DIR/<source>/train.jsonl)
"""
import json, os, random, sys
from datasets import load_dataset

R = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("CART_ROOT", "."), "data", "neg")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
r = random.Random(0)


def write(name, rows):
    rows = [(p, t) for p, t in rows if p.strip() and t.strip() and len(p) < 12_000]
    r.shuffle(rows)
    os.makedirs(f"{R}/{name}", exist_ok=True)
    with open(f"{R}/{name}/train.jsonl", "w") as f:
        for i, (p, t) in enumerate(rows[:N]):
            f.write(json.dumps({"id": f"neg-{name}-{i}", "prompt": p, "target": " " + t.strip(), "gold": None}) + "\n")
    print(f"{name}: {min(N, len(rows))} prompts, median prompt {sorted(len(p) for p, _ in rows[:N])[min(N, len(rows)) // 2]} chars", flush=True)


def take(ds, k=2000):
    return [ds[i] for i in r.sample(range(len(ds)), min(k, len(ds)))]


def src_dolly():
    d = load_dataset("databricks/databricks-dolly-15k", split="train")
    write("dolly", [((x["context"] + "\n\n" if x["context"] else "") + x["instruction"] + "\nResponse:", x["response"]) for x in take(d)])


def src_alpaca():
    d = load_dataset("tatsu-lab/alpaca", split="train")
    write("alpaca", [(x["instruction"] + ("\n" + x["input"] if x["input"] else "") + "\nResponse:", x["output"]) for x in take(d)])


def src_gsm8k():
    d = load_dataset("openai/gsm8k", "main", split="train")
    write("gsm8k", [("Question: " + x["question"] + "\nAnswer:", x["answer"]) for x in take(d)])


def src_squad():
    d = load_dataset("rajpurkar/squad", split="train")
    write("squad", [(x["context"] + "\n\nQuestion: " + x["question"] + "\nAnswer:", x["answers"]["text"][0]) for x in take(d)])


def src_mbpp():
    d = load_dataset("google-research-datasets/mbpp", "full", split="train")
    write("mbpp", [(x["text"] + "\nWrite a Python function.\n```python\n", x["code"]) for x in take(d)])


def src_agnews():
    d = load_dataset("fancyzhx/ag_news", split="train")
    lab = ["World", "Sports", "Business", "Sci/Tech"]
    write("agnews", [("Article: " + x["text"] + "\nTopic (World, Sports, Business or Sci/Tech):", lab[x["label"]]) for x in take(d)])


def src_imdb():
    d = load_dataset("stanfordnlp/imdb", split="train")
    write("imdb", [("Review: " + x["text"][:3000] + "\nSentiment (positive or negative):", "positive" if x["label"] else "negative") for x in take(d)])


def src_wikitext():
    d = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    paras = [x["text"].strip() for x in d if len(x["text"]) > 400 and not x["text"].strip().startswith("=")]
    write("wikitext", [("\n\n".join(paras[i:i + 3]) + "\n\n", paras[i + 3][:500]) for i in range(0, len(paras) - 4, 4)])


for fn in (src_dolly, src_alpaca, src_gsm8k, src_squad, src_mbpp, src_agnews, src_imdb, src_wikitext):
    try:
        fn()
    except Exception as e:                      # a source that will not load is skipped, loudly
        print(f"SKIPPED {fn.__name__}: {type(e).__name__}: {str(e)[:160]}", flush=True)
print("sources written:", sorted(os.listdir(R)))
