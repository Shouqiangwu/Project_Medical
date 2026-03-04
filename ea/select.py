import random
from typing import List, Tuple

import numpy as np

from ea.individual import Individual


def rank_selection_probs(fitness: List[float], temperature: float = 1.0) -> np.ndarray:
    """
    Softmax-based selection probabilities. Allows negative fitness.
    """
    f = np.array(fitness, dtype=np.float64)
    f = f - np.max(f)
    f = f / max(1e-8, temperature)
    w = np.exp(f)
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        return np.ones_like(w) / len(w)
    return w / w_sum


def select_parents(
    pop: List[Individual],
    num_pairs: int,
    temperature: float,
    rng: random.Random,
) -> List[Tuple[Individual, Individual]]:
    """
    Roulette-wheel / softmax parent selection. Returns (parentA, parentB) pairs.
    """
    fitness = [ind.fitness for ind in pop]
    probs = rank_selection_probs(fitness, temperature=temperature)
    cdf = np.cumsum(probs)

    pairs = []
    for _ in range(num_pairs):
        i = int(np.searchsorted(cdf, rng.random(), side="right"))
        j = int(np.searchsorted(cdf, rng.random(), side="right"))
        i = min(max(i, 0), len(pop) - 1)
        j = min(max(j, 0), len(pop) - 1)
        pairs.append((pop[i], pop[j]))
    return pairs
