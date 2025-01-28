from typing import TypeVar
from fuzzer.common.seed_wrapper import SeedWrapper

SeedWrapperClass = TypeVar('SeedWrapperClass', bound=SeedWrapper)

class RandomFitness:

    def __init__(self,
                 refer_seed: SeedWrapperClass,
                 mode: str,
                 line_unit: float):
        self.refer_seed = refer_seed
        self.mode = mode # [behavior, path, both]
        self.line_unit = line_unit

    def calculate(self, seed: SeedWrapperClass) -> float:
        seed.update_record()
        return 1.0