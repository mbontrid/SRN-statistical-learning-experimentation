import random
from collections.abc import Generator
from typing import Any


def rand_head_emb_tail_generator(
    head_tail: list[Any, Any] | tuple[Any, Any],
    embeddings: list | tuple,
    seed: int | None = None,
) -> Generator[tuple[Any, Any, Any], None, None]:
    """Generate a random "key", "embeddings", "value" tuple.

    Args:
        head_tail: list of tuples containing the head and tail values.
        embeddings: list of available embeddings.
        seed: random seed for reproducibility.

    Returns:
        tuple: A tuple containing the first key, a randomly selected embedding, and the last value.
    """
    if seed is not None:
        random.seed(seed)
    while True:
        first, last = random.choice(list(head_tail))
        embedding = random.choice(embeddings)
        yield from (first, embedding, last)
