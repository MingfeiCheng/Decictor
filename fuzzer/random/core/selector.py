import copy
import random

from typing import List

from fuzzer.common.seed_wrapper import SeedWrapper


class RandomSelector:
    """
    Max is better
    """

    def __init__(self, population_size: int):
        self.population_size = population_size

    def _random_select(self, corpus: List[SeedWrapper]) -> List[SeedWrapper]:
        next_parent = list()
        for i in range(self.population_size):
            select = random.choice(corpus)
            next_parent.append(copy.deepcopy(select))
        return next_parent

    def select(self, corpus: List[SeedWrapper]) -> List[SeedWrapper]:
        return self._random_select(corpus)