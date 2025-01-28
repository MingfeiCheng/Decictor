from typing import TypeVar
from fuzzer.common.seed_wrapper import SeedWrapper

SeedWrapperClass = TypeVar('SeedWrapperClass', bound=SeedWrapper)

class FitnessCalculator:

    def __init__(self):
        pass

    def calculate(self, seed: SeedWrapperClass) -> float:
        # TODO: add acceleration
        # only consider: collision and reach destination
        basic_result = seed.result

        feedback = basic_result['feedback']

        min_dist = float(feedback['collision'])
        # dist2dest = float(feedback['destination'])
        ego_attributes = seed.record.ego.attributes
        total_frame = len(ego_attributes)
        good_count = 0
        for item in ego_attributes:
            speed = item[3]
            acc = item[4]
            if abs(acc) <= 0.6:
                good_count += 1

        # norm dist
        # min_dist = float(min(min_dist, 1.0))
        # dist2dest = float(max((20 - dist2dest) / 20.0, 0.0))
        fitness = min_dist + float(good_count) / float(total_frame)
        # fitness = float((min_dist + dist2dest) / 2.0)
        return fitness