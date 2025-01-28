import os
import copy

from omegaconf import DictConfig
from loguru import logger
from datetime import datetime

from tools.result_recorder import ResultRecorder
from tools.global_config import GlobalConfig
from fuzzer.common.seed_wrapper import SeedWrapper
from fuzzer.decictor.core.oracle import OptimalOracle

from fuzzer.deep_collision.network import DQN, HyperParameter
from scenario.runner.scenario_runner_RL import ScenarioRunner

class DeepCollision(object):

    def __init__(
            self,
            cfg_fuzzer: DictConfig,
            cfg_runner: DictConfig
    ):

        self.policy = DQN()

        self.save_root = GlobalConfig.save_dir
        self.run_hour = cfg_fuzzer.run_hour
        self.seed_id = 0
        # load seed
        refer_seed_json = os.path.join(GlobalConfig.seed_dir, 'result/result_0.json')
        if not os.path.isfile(refer_seed_json):
            raise RuntimeError(f'{refer_seed_json} not exists')
        self.refer_seed: SeedWrapper = SeedWrapper.from_json(refer_seed_json)
        self.refer_seed.id = self.seed_id
        self.refer_seed.save_root = self.save_root  # !!!
        self.seed_id += 1

        # load scenario runner
        self.scenario_runner = ScenarioRunner(cfg_runner, self.policy)

        self.optimal_checker = OptimalOracle(
            self.refer_seed,
            cfg_fuzzer.optimal_threshold,
            cfg_fuzzer.grid_unit
        )
        self.result_recorder = ResultRecorder(GlobalConfig.save_dir)

        self.safe_violations = list()
        self.mutate_violations = list()
        self.nods_violations = list()
        logger.info('Loaded DeepCollision')

    @staticmethod
    def _terminal_check(start_time: datetime, run_hour: float) -> bool:
        curr_time = datetime.now()
        t_delta = (curr_time - start_time).total_seconds()
        if t_delta / 3600.0 > run_hour:
            logger.info(f'Finish fuzzer as reach the time limited: {t_delta / 3600.0}/{run_hour}')
            return True
        return False

    def run(self):
        start_time = datetime.now()
        while True:
            if self._terminal_check(start_time, self.run_hour):
                return

            mutated_seed = copy.deepcopy(self.refer_seed)
            mutated_seed.update_id(self.seed_id)
            self.seed_id += 1
            # start runner
            mutated_seed: SeedWrapper = self.scenario_runner.run(mutated_seed)

            logger.debug(f"memory_counter: {self.policy.memory_counter}")
            if self.policy.memory_counter > HyperParameter['MEMORY_SIZE']:
                feedback_start_time = datetime.now()
                self.policy.learn()
                feedback_end_time = datetime.now()
                logger.info('--> [Simulation Time] Feedback Spend Time: [=]{}[=]', (feedback_end_time - feedback_start_time).total_seconds())

            mr_oracle = self.optimal_checker.check(mutated_seed)
            fitness = 0.0
            mutated_seed.update_oracle(oracle=mr_oracle)
            mutated_seed.update_fitness(fitness)
            mutated_seed.save()

            self.result_recorder.add_item(mutated_seed.record_item)
            self.result_recorder.write_file(
                mutated_seed.record_columns)  # write script to parse result to obtain details

            # optimal
            # 6. check
            is_pass_replay = mr_oracle[0]
            is_non_optimal = mr_oracle[1]
            is_safe = len(mutated_seed.result['violation']) <= 0
            # - check rollback
            if not is_pass_replay:
                # trigger rollback
                self.mutate_violations.append(mutated_seed.id)

            if not is_safe:
                # trigger safe violation
                self.safe_violations.append(mutated_seed.id)
            else:
                # check mr
                if is_non_optimal and is_pass_replay:
                    # trigger mr
                    self.nods_violations.append(mutated_seed.id)

            # record violations
            # save existing ids
            violation_id_file = os.path.join(self.save_root, 'violation_ids.csv')
            with open(violation_id_file, 'w') as f:
                for idx in self.nods_violations:
                    f.write(f"nods, {idx}\n")

                for idx in self.safe_violations:
                    f.write(f"safe, {idx}\n")

                for idx in self.mutate_violations:
                    f.write(f"mutate, {idx}\n")