import pytest
import torch

from adapters.lora import LoRA
from conftest import as_decoder, tiny_model
from shared import dist
from shared.adapter import Adapter, AdapterKind
from shared.backprop import BackpropCfg, logprobs
from test_backprop import kv_adapter, make_rows


def groups():
    out = []
    for g in range(4):
        rows = make_rows(seed=10 + g)
        w = torch.linspace(-1, 1, len(rows)).tolist()
        out.append((rows, w))
    return out


def job(adapter, cfg):
    def fn(dec, rw):
        rows, w = rw
        got = logprobs(dec, adapter, rows, [torch.tensor(r.ids)[:, None] for r in rows], cfg)
        (-sum(wi * g.sum() for wi, g in zip(w, got))).backward()
    return fn


@pytest.mark.parametrize("kind", ["kv_prefix", "lora"])
def test_replicas_sum_to_single(kind):
    decs = [as_decoder(tiny_model()) for _ in range(3)]
    if kind == "lora":
        mod = LoRA(decs[0].model, 1, 32.0, ("q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"), (), 7, False)
        with torch.no_grad():
            for b in mod.B:
                b.normal_(0, 0.05)
        a = Adapter(AdapterKind.lora, mod)
    else:
        a = kv_adapter(decs[0])
    cfg = BackpropCfg()
    gs = groups()
    dist.score_parallel(decs[:1], gs, job(a, cfg))
    one = [p.grad.clone() for p in a.params()]
    for p in a.params():
        p.grad = None
    dist.score_parallel(decs, gs, job(a, cfg))
    three = [p.grad.clone() for p in a.params()]
    rel = max(float((x - y).norm() / (x.norm() + 1e-30)) for x, y in zip(one, three))
    assert sum(float(x.norm()) for x in one) > 0 and rel < 1e-5, rel
