from typing import TypeVar, Tuple
from fuzzer.common.seed_wrapper import SeedWrapper

SeedWrapperClass = TypeVar('SeedWrapperClass', bound=SeedWrapper)

class FitnessCalculator:

    def __init__(self):
        pass

    def calculate(self, seed: SeedWrapperClass) -> Tuple[float, float, float]:
        # only consider: collision and reach destination
        basic_result = seed.result

        feedback = basic_result['feedback']

        min_dist = float(feedback['collision'])
        dist2dest = float(feedback['destination'])

        # norm dist
        min_dist = float(min(min_dist, 1.0))
        dist2dest = float(max((20 - dist2dest) / 20.0, 0.0))


        return min_dist, dist2dest, (min_dist + dist2dest) / 2.0