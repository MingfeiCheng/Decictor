import os
import copy
import numpy as np

from omegaconf import DictConfig
from loguru import logger
from datetime import datetime

from tools.config_utils import load_entry_point
from tools.result_recorder import ResultRecorder
from tools.local_logger import get_instance_logger
from tools.global_config import GlobalConfig
from fuzzer.decictor.core.oracle import OptimalOracle
from fuzzer.common.seed_wrapper import SeedWrapper

from fuzzer.drivefuzz.core.mutator import ScenarioMutator
from fuzzer.drivefuzz.core.fitness import FitnessCalculator

from scenario.runner.scenario_runner import ScenarioRunner

class DriveFuzzer:
    """
    Randomly generate initial scenario and evaluate it.
    """

    def __init__(
            self,
            cfg_fuzzer: DictConfig,
            cfg_runner: DictConfig
    ):

        self.save_root = GlobalConfig.save_dir

        self.run_hour = cfg_fuzzer.run_hour
        self.curr_iteration = 0
        self.best_seed = None

        # result recorder
        self.result_recorder = ResultRecorder(GlobalConfig.save_dir)

        # config runner
        self.scenario_runner = ScenarioRunner(cfg_runner)

        # config mutator
        self.scenario_mutator = ScenarioMutator(cfg_fuzzer.mutator)

        # config fitness
        self.fitness_calculator = FitnessCalculator() # NOTE: Min is better

        # config optimal oracle
        # load refer seed - the input is the optimal seed. only one I think
        self.seed_id = 0
        refer_seed_json = os.path.join(cfg.seed_folder, 'result/result_0.json')
        if not os.path.isfile(refer_seed_json):
            logger.debug(refer_seed_json)
            raise RuntimeError(f'{refer_seed_json} not exists')
        self.refer_seed: SeedWrapper = SeedWrapper.from_json(refer_seed_json)
        self.refer_seed.id = self.seed_id
        self.refer_seed.save_root = self.save_root # !!!
        self.seed_id += 1
        self.optimal_checker = OptimalOracle(self.refer_seed, cfg.fuzzer.optimal_threshold, cfg.fuzzer.grid_unit)

        debug_folder = os.path.join(self.save_root, 'debug')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        fitness_log_file = os.path.join(debug_folder, f"fitness.log")
        mutation_log_file = os.path.join(debug_folder, f"mutation.log")
        if os.path.isfile(fitness_log_file):
            os.remove(fitness_log_file)
        if os.path.isfile(mutation_log_file):
            os.remove(mutation_log_file)

        self.logger_fitness = get_instance_logger(f"fitness", fitness_log_file)
        self.logger_mutation = get_instance_logger("mutation", mutation_log_file)

        self.logger_fitness.info("Logger initialized for fitness")
        self.logger_mutation.info("Logger initialized for mutation")

        self.safe_violations = list()
        self.mutate_violations = list()
        self.nods_violations = list()

    @staticmethod
    def _terminal_check(start_time: datetime, run_hour: float) -> bool:
        curr_time = datetime.now()
        t_delta = (curr_time - start_time).total_seconds()
        if t_delta / 3600.0 > run_hour:
            logger.info(f'Finish fuzzer as reach the time limited: {t_delta / 3600.0}/{run_hour}')
            return True
        return False

    def run(self):
        # minimize is better
        logger.info('===== Start Fuzzer (DriveFuzz) =====')
        # generate initial scenario

        start_time = datetime.now()
        while True:
            if self.best_seed is None:
                self.best_seed = copy.deepcopy(self.refer_seed)
                self.best_seed = self.scenario_mutator.get_random_traffic(self.best_seed)

            if self._terminal_check(start_time, self.run_hour):
                return

            source_seed = copy.deepcopy(self.best_seed)

            # TODO: Check this
            mutated_seed, _ = self.scenario_mutator.mutate(source_seed)
            mutated_seed.update_id(self.seed_id)
            self.seed_id += 1
            mutated_seed: SeedWrapper = self.scenario_runner.run(mutated_seed)
            # 4. calculate information for both (actual we do not need this)
            mr_oracle = self.optimal_checker.check(mutated_seed)
            fitness = self.fitness_calculator.calculate(mutated_seed)
            mutated_seed.update_oracle(oracle=mr_oracle)
            mutated_seed.update_fitness(fitness)
            mutated_seed.save()
            self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")
            self.logger_mutation.info(f"{mutated_seed.id},{source_seed.id}")

            self.result_recorder.add_item(mutated_seed.record_item)
            self.result_recorder.write_file(mutated_seed.record_columns)  # write script to parse result to obtain details

            m_start_time = datetime.now()
            m_end_time = datetime.now()
            logger.info('Energy Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())

            # calculate the diversity based on the record
            m_start_time = datetime.now()
            if mutated_seed.fitness < source_seed.fitness:
                self.best_seed = copy.deepcopy(mutated_seed)

            m_end_time = datetime.now()
            logger.info('Coverage Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())

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
                self.best_seed = None # Found violation
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
