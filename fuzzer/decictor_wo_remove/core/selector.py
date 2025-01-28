import copy
from typing import List

import numpy as np
from omegaconf import DictConfig

from fuzzer.common.seed_wrapper import SeedWrapper


class DecictorSelector:
    """
    Max is better
    """

    def __init__(self, cfg: DictConfig, population_size: int):
        self.population_size = population_size
        self.select_mode = cfg.mode

    def _roulette_select(self, corpus: List[SeedWrapper]) -> List[SeedWrapper]:
        # fitness max is better, optimize with max
        tmp_fitness = list()
        for i in range(len(corpus)):
            tmp_p_i_fitness = corpus[i].fitness
            tmp_p_i_fitness = np.clip(tmp_p_i_fitness, 1e-5, None)  # max is better, but need to be 0.0 - 1.0
            tmp_fitness.append(tmp_p_i_fitness)

        # norm
        def equal_check(values):
            v_min = min(values)
            v_max = max(values)
            if v_max == v_min:
                # Handle the case where all values are the same, if needed
                return [0.5 for _ in values]  # or return values if you want to keep them unchanged
            return values

        tmp_fitness = equal_check(tmp_fitness)
        fitness_sum = float(sum(tmp_fitness))
        probabilities = np.array([(tmp_f / fitness_sum) for tmp_f in tmp_fitness])

        next_parent = list()
        max_prob_index = np.argmax(probabilities)
        best_one = corpus[max_prob_index]
        next_parent.append(copy.deepcopy(best_one))
        for i in range(self.population_size - 1):
            select = np.random.choice(corpus, p=probabilities)
            next_parent.append(copy.deepcopy(select))
        return next_parent

    def select(self, corpus: List[SeedWrapper]) -> List[SeedWrapper]:
        if self.select_mode.lower() == 'roulette':
            return self._roulette_select(corpus)
        else:
            raise KeyError