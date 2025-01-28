import os
import copy
import numpy as np

from omegaconf import DictConfig
from loguru import logger
from datetime import datetime

from tools.result_recorder import ResultRecorder
from tools.local_logger import get_instance_logger
from tools.global_config import GlobalConfig

from fuzzer.decictor.core.oracle import OptimalOracle
from fuzzer.common.seed_wrapper import SeedWrapper

from fuzzer.behavexplor.models.coverage import CoverageModel
from fuzzer.behavexplor.core.mutator import ScenarioMutator
from fuzzer.behavexplor.core.fitness import FitnessCalculator

from scenario.runner.scenario_runner import ScenarioRunner


class BehAVExplorFuzzer:
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
        self.window_size = cfg_fuzzer.window_size
        self.cluster_num = cfg_fuzzer.cluster_num
        self.threshold_coverage = cfg_fuzzer.threshold_coverage # 0.4
        self.threshold_energy = cfg_fuzzer.threshold_energy
        self.feature_resample = cfg_fuzzer.feature_resample # 'linear'
        self.initial_corpus_size = cfg_fuzzer.initial_corpus_size

        self.coverage_model = CoverageModel(
            self.window_size,
            self.cluster_num,
            self.threshold_coverage
        )

        self.curr_iteration = 0
        self.corpus = list()  # save all elements in the fuzzing
        self.corpus_energy = list()
        self.corpus_fail = list()
        self.corpus_mutation = list()

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
        refer_seed_json = os.path.join(GlobalConfig.seed_dir, 'result/result_0.json')
        if not os.path.isfile(refer_seed_json):
            logger.debug(refer_seed_json)
            raise RuntimeError(f'{refer_seed_json} not exists')
        self.refer_seed: SeedWrapper = SeedWrapper.from_json(refer_seed_json)
        self.refer_seed.id = self.seed_id
        self.refer_seed.save_root = self.save_root # !!!
        self.seed_id += 1
        self.optimal_checker = OptimalOracle(
            self.refer_seed,
            cfg_fuzzer.optimal_threshold,
            cfg_fuzzer.grid_unit
        )

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

    def _seed_selection(self):
        select_probabilities = copy.deepcopy(self.corpus_energy)
        select_probabilities = np.array(select_probabilities) + 1e-5
        select_probabilities /= (select_probabilities.sum())
        source_seed_index = np.random.choice(list(np.arange(0, len(self.corpus))), p=select_probabilities)
        return source_seed_index

    def run(self):
        # minimize is better
        logger.info('===== Start Fuzzer (BehAVExplor) =====')
        # generate initial scenario
        while len(self.corpus) < self.initial_corpus_size:
            mutated_seed = self.scenario_mutator.get_random_traffic(self.refer_seed)
            mutated_seed.update_id(self.seed_id)
            self.seed_id += 1
            mutated_seed: SeedWrapper = self.scenario_runner.run(mutated_seed)
            mr_oracle = self.optimal_checker.check(mutated_seed)
            fitness = self.fitness_calculator.calculate(mutated_seed)
            mutated_seed.update_oracle(oracle=mr_oracle)
            mutated_seed.update_fitness(fitness)
            mutated_seed.save()

            self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")
            self.logger_mutation.info(f"{mutated_seed.id},{mutated_seed.id},initial,1.0")
            self.result_recorder.add_item(mutated_seed.record_item)
            self.result_recorder.write_file(mutated_seed.record_columns)  # write script to parse result to obtain details

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

            violation_id_file = os.path.join(self.save_root, 'violation_ids.csv')
            with open(violation_id_file, 'w') as f:
                for idx in self.nods_violations:
                    f.write(f"nods, {idx}\n")

                for idx in self.safe_violations:
                    f.write(f"safe, {idx}\n")

                for idx in self.mutate_violations:
                    f.write(f"mutate, {idx}\n")

            if len(mutated_seed.result['violation']) > 0:
                continue

            # check conditions
            self.corpus.append(copy.deepcopy(mutated_seed))
            # the following items are used for energy
            self.corpus_fail.append(0)
            self.corpus_mutation.append(0)
            self.corpus_energy.append(1)

        # initialize the coverage model
        if len(self.corpus) > 0:
            m_start_time = datetime.now()
            self.coverage_model.initialize(self.corpus)
            m_end_time = datetime.now()
            logger.info('Coverage Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())

        start_time = datetime.now()
        while True:
            if self._terminal_check(start_time, self.run_hour):
                return

            source_seed_index = self._seed_selection() # select based on energy, high energy is better
            source_seed = self.corpus[source_seed_index]
            source_seed_energy = self.corpus_energy[source_seed_index]
            source_seed_fail = self.corpus_fail[source_seed_index]
            source_seed_mutation = self.corpus_mutation[source_seed_index]

            if source_seed_energy > self.threshold_energy:
                mutation_stage = 'small'
            else:
                mutation_stage = 'large'

            mutated_seed, _ = self.scenario_mutator.mutate(source_seed, mutation_stage)
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
            self.logger_mutation.info(f"{mutated_seed.id},{source_seed.id},{mutation_stage},{source_seed_energy}")

            self.result_recorder.add_item(mutated_seed.record_item)
            self.result_recorder.write_file(mutated_seed.record_columns)  # write script to parse result to obtain details

            m_start_time = datetime.now()

            # add mutation
            source_seed_mutation += 1

            # update energy & fail
            benign = True
            if len(mutated_seed.result['violation']) > 0:
                source_seed_fail += 1
                benign = False

            if mutation_stage == 'large':
                source_seed_energy = source_seed_energy - 0.15
            else:
                # update energy of source_seed
                delta_fail = source_seed_fail / float(source_seed_mutation)
                if benign:
                    delta_fail = min(delta_fail, 1.0)
                    delta_fail = -0.5 * (1 - delta_fail)
                    # delta_fail = -0.1 * (1 - delta_fail)
                else:
                    delta_fail = min(delta_fail, 1.0)

                delta_fitness = source_seed.fitness - mutated_seed.fitness #
                delta_select = -0.1
                source_seed_energy = source_seed_energy  + 0.5 * delta_fail + 0.3 * np.tanh(delta_fitness) + delta_select

            # update information
            self.corpus_energy[source_seed_index] = float(np.clip(source_seed_energy, 1e-5, 4.0))
            self.corpus_fail[source_seed_index] = source_seed_fail
            self.corpus_mutation[source_seed_index] = source_seed_mutation
            m_end_time = datetime.now()
            logger.info('Energy Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())

            # calculate the diversity based on the record
            m_start_time = datetime.now()
            follow_up_seed_is_new, follow_up_seed_div, follow_up_seed_ab = self.coverage_model.feedback_coverage_behavior(mutated_seed)
            if follow_up_seed_is_new or mutated_seed.fitness < source_seed.fitness:
                self.corpus.append(copy.deepcopy(mutated_seed))
                self.corpus_fail.append(0)
                self.corpus_mutation.append(0)
                self.corpus_energy.append(1)
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
