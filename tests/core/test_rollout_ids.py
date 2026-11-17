import pytest
from transformers import AutoTokenizer

from shared.rollout import rollout_ids


@pytest.mark.parametrize("name", ["Qwen/Qwen3-30B-A3B-Base"])
def test_rollout_ids(name):
    t = AutoTokenizer.from_pretrained(name)
    text = " The numbers are not necessarily equal, so $a=24$."
    canon = t.encode(text, add_special_tokens=False)
    j = max(range(len(canon)), key=lambda i: len(t.decode([canon[i]])))
    piece = t.decode([canon[j]])
    split = [x for ch in piece for x in t.encode(ch, add_special_tokens=False)]
    if t.decode(canon[:j] + split + canon[j + 1:]) != text:
        split = t.encode(piece[:2], add_special_tokens=False) + t.encode(piece[2:], add_special_tokens=False)
    ids = canon[:j] + split + canon[j + 1:]
    assert ids != canon
    stop = t.encode("\nProblem:", add_special_tokens=False)
    r, txt = rollout_ids(t, text, ids + stop, 640, False)
    assert r == ids and txt == text and t.decode(r) == text, "non-canonical ids kept, stop trimmed"
    assert rollout_ids(t, text, ids + stop, 640, True) == (ids + stop, text), "score_stop keeps the stop tokens"
    other = t.encode("unrelated", add_special_tokens=False)
    assert rollout_ids(t, text, other, 640, False) == (other, t.decode(other)), "text follows the engine's ids"
    r5, t5 = rollout_ids(t, text, ids + stop, 5, False)
    assert r5 == ids[:5] and t5 == t.decode(ids[:5]), "capped at max_tokens, text follows"
