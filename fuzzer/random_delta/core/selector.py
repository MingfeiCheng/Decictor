import copy
import random
from typing import List

import numpy as np
from omegaconf import DictConfig

from fuzzer.common.seed_wrapper import SeedWrapper


class RandomSelector:
    """
    Max is better
    """

    def __init__(self, cfg: DictConfig, population_size: int):
        self.population_size = population_size
        self.select_mode = cfg.mode

    def _roulette_select(self, corpus: List[SeedWrapper]) -> List[SeedWrapper]:
        next_parent = list()
        # max_prob_index = np.argmax(probabilities)
        # best_one = corpus[max_prob_index]
        # next_parent.append(copy.deepcopy(best_one))
        for i in range(self.population_size):
            # select = np.random.choice(corpus, p=probabilities)
            select = random.choice(corpus)
            next_parent.append(copy.deepcopy(select))
        return next_parent

    def select(self, corpus: List[SeedWrapper]) -> List[SeedWrapper]:
        if self.select_mode.lower() == 'roulette':
            return self._roulette_select(corpus)
        else:
            raise KeyError