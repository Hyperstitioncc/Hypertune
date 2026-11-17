from types import SimpleNamespace


def get_current_vllm_config():
    return SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=4))
