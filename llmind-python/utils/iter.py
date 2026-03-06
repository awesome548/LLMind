from __future__ import annotations

from typing import Generator, Sequence, TypeVar

T = TypeVar("T")


def chunked(sequence: Sequence[T], size: int) -> Generator[Sequence[T], None, None]:
    if size <= 0:
        raise ValueError("chunk size must be greater than 0")
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]
