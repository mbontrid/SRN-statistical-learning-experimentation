import random
from typing import Any, Generator


def rand_key_emb_value(
    seq_stimulus: dict, embeddings: list | tuple, seed: int | None = None
) -> Generator[Any, None, None]:
    """Generate a random "key", "embeddings", "value" tuple.

    Args:
        seq_stumulus: dictionary of key to put at the beginnig and value to put at the end after the embedding.
        embeddings: list of available embeddings.
        seed: random seed for reproducibility.

    Returns:
        tuple: A tuple containing the first key, a randomly selected embedding, and the last value.
    """
    if seed is not None:
        random.seed(seed)
    while True:
        first, last = random.choice(list(seq_stimulus.items()))
        embedding = random.choice(embeddings)
        yield from (first, embedding, last)
