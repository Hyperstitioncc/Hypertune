from __future__ import annotations


def check(splits: tuple[str, ...], split: str) -> None:
    if split not in splits:
        raise ValueError(f"unknown split {split!r}; valid splits: {', '.join(splits)}")
