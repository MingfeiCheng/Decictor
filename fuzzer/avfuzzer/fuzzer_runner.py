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

from fuzzer.avfuzzer.core.mutator import ScenarioMutator
from fuzzer.avfuzzer.core.fitness import FitnessCalculator

from scenario.runner.scenario_runner import ScenarioRunner

class AVFuzzer:
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
        self.local_run_hour = 0.5

        self.best_seed = None
        self.best_fitness_after_restart = 10
        self.best_fitness_lst = []

        self.pm = 0.6
        self.pc = 0.6

        self.minLisGen = 5  # Min gen to start LIS
        self.curr_population = list()
        self.prev_population = list()
        self.curr_iteration = 0
        self.last_restart_iteration = 0
        self.population_size = 4

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
        if os.path.isfile(fitness_log_file):
            os.remove(fitness_log_file)

        self.logger_fitness = get_instance_logger(f"fitness", fitness_log_file)
        self.logger_fitness.info("Logger initialized for fitness")

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

    def _seed_selection(self, curr_population, prev_population):
        # fitness -> min is better
        tmp_population = curr_population + prev_population

        tmp_fitness = list()
        for i in range(len(tmp_population)):
            tmp_p_i_fitness = tmp_population[i].fitness
            tmp_fitness.append(tmp_p_i_fitness + 1e-5)

        tmp_fitness_sum = float(sum(tmp_fitness))
        tmp_probabilities = np.array([(tmp_f / tmp_fitness_sum) for tmp_f in tmp_fitness])
        tmp_probabilities = 1 - np.array(tmp_probabilities)
        tmp_probabilities /= tmp_probabilities.sum()

        next_parent = list()
        # next_parent = [copy.deepcopy(self.best_seed)]
        for i in range(self.population_size):
            select = np.random.choice(tmp_population, p=tmp_probabilities)
            next_parent.append(copy.deepcopy(select))

        return next_parent

    def _run_global(self, start_time):
        # minimize is better
        logger.info('===== Start Fuzzer (AVFuzzer) =====')

        # initial stage
        self.best_seed = None
        self.best_fitness_after_restart = 10
        self.best_fitness_lst = []

        self.pm = 0.6
        self.pc = 0.6

        self.minLisGen = 5  # Min gen to start LIS
        self.curr_population = list()
        self.prev_population = list()
        self.curr_iteration = 0
        self.last_restart_iteration = 0
        self.population_size = 4

        initial_seed = self.scenario_mutator.get_initial_traffic(self.refer_seed)
        while len(self.curr_population) < self.population_size:
            mutated_seed = self.scenario_mutator.initial_mutate(initial_seed)
            mutated_seed.update_id(self.seed_id)
            self.seed_id += 1
            mutated_seed: SeedWrapper = self.scenario_runner.run(mutated_seed)
            mr_oracle = self.optimal_checker.check(mutated_seed)
            fitness = self.fitness_calculator.calculate(mutated_seed)
            mutated_seed.update_oracle(oracle=mr_oracle)
            mutated_seed.update_fitness(fitness)
            mutated_seed.save()

            self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")
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
                if 'collision' in mutated_seed.result['violation']:
                    return
                else:
                    continue

            self.curr_population.append(copy.deepcopy(mutated_seed))

        noprogress = False
        while True:  # i th generation.
            if self._terminal_check(start_time, self.run_hour):
                return

            self.curr_iteration += 1
            logger.info('===== Iteration {} =====', self.curr_iteration)

            if not noprogress:
                if self.curr_iteration == 1:
                    self.prev_population = copy.deepcopy(self.curr_population)
                else:
                    self.prev_population = self._seed_selection(self.curr_population, self.prev_population)
                # mutation
                self.curr_population = self.scenario_mutator.mutate(self.prev_population, self.pc, self.pm)
            else:
                # restart
                initial_seed = self.scenario_mutator.get_initial_traffic(self.refer_seed)
                for i in range(self.population_size):
                    self.curr_population[i] = self.scenario_mutator.initial_mutate(initial_seed)
                self.best_seed = None

            # run
            for i in range(self.population_size):
                if self._terminal_check(start_time, self.run_hour):
                    return

                curr_seed = self.curr_population[i]
                curr_seed.update_id(self.seed_id)
                self.seed_id += 1
                curr_seed: SeedWrapper = self.scenario_runner.run(curr_seed)
                mr_oracle = self.optimal_checker.check(curr_seed)
                fitness = self.fitness_calculator.calculate(curr_seed)
                curr_seed.update_oracle(oracle=mr_oracle)
                curr_seed.update_fitness(fitness)
                curr_seed.save()

                self.logger_fitness.info(f"{curr_seed.id},{curr_seed.fitness}")
                self.result_recorder.add_item(curr_seed.record_item)
                self.result_recorder.write_file(curr_seed.record_columns)  # write script to parse result to obtain details

                # optimal
                # 6. check
                is_pass_replay = mr_oracle[0]
                is_non_optimal = mr_oracle[1]
                is_safe = len(curr_seed.result['violation']) <= 0
                # - check rollback
                if not is_pass_replay:
                    # trigger rollback
                    self.mutate_violations.append(curr_seed.id)

                if not is_safe:
                    # trigger safe violation
                    self.safe_violations.append(curr_seed.id)
                else:
                    # check mr
                    if is_non_optimal and is_pass_replay:
                        # trigger mr
                        self.nods_violations.append(curr_seed.id)

                # check conditions
                if not is_safe:
                    logger.info('Find violation, exit fuzzer.') # todo: restart
                    return

                self.curr_population[i] = curr_seed
                if self.best_seed is None or curr_seed.fitness < self.best_seed.fitness:
                    self.best_seed = copy.deepcopy(curr_seed)

            self.best_fitness_lst.append(self.best_seed.fitness)
            if noprogress:
                self.best_fitness_after_restart = self.best_seed.fitness
                noprogress = False

            # check progress with previous 5 fitness
            ave = 0
            if self.curr_iteration >= self.last_restart_iteration + 5:
                for j in range(self.curr_iteration - 5, self.curr_iteration):
                    ave += self.best_fitness_lst[j]
                ave /= 5
                if ave <= self.best_seed.fitness:
                    self.last_restart_iteration = self.curr_iteration
                    noprogress = True

            #################### End the Restart Process ###################
            if self.best_seed.fitness < self.best_fitness_after_restart:
                if self.curr_iteration > (self.last_restart_iteration + self.minLisGen):  # Only allow one level of recursion
                    ################## Start LIS #################
                    lis_best_seed, find_bug = self._run_local(start_time)
                    if find_bug:
                        logger.info('Find violation or timeout, exit fuzzer.') # todo restart
                        return

                    if lis_best_seed.fitness < self.best_seed.fitness:
                        self.curr_population[0] = copy.deepcopy(lis_best_seed)
                    logger.info(' === End of Local Iterative Search === ')


    def _run_local(self, start_time):

        local_start_time = datetime.now()

        local_pm = 0.6 * 1.5
        local_pc = 0.6 * 1.5

        curr_population = list()
        prev_population = list()

        local_best = copy.deepcopy(self.best_seed)

        logger.info('===== Start Local (AVFuzzer) =====')
        # generate initial scenario
        prev_population = [copy.deepcopy(local_best) for _ in range(self.population_size)]
        curr_population = self.scenario_mutator.mutate(prev_population, local_pc, local_pm)
        for i in range(self.population_size):

            if self._terminal_check(start_time, self.run_hour) or self._terminal_check(local_start_time, self.local_run_hour):
                return copy.deepcopy(local_best), False

            mutated_seed = curr_population[i]
            mutated_seed.update_id(self.seed_id)
            self.seed_id += 1
            mutated_seed: SeedWrapper = self.scenario_runner.run(mutated_seed)
            mr_oracle = self.optimal_checker.check(mutated_seed)
            fitness = self.fitness_calculator.calculate(mutated_seed)
            mutated_seed.update_oracle(oracle=mr_oracle)
            mutated_seed.update_fitness(fitness)
            mutated_seed.save()

            self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")
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

            # check conditions
            if not is_safe:
                logger.info('Find violation, exit fuzzer.') # TODO: check this
                return None, True

            curr_population[i] = mutated_seed

        while True:  # i th generation.
            if self._terminal_check(start_time, self.run_hour) or self._terminal_check(local_start_time, self.local_run_hour):
                return copy.deepcopy(local_best), False

            prev_population = self._seed_selection(curr_population, prev_population)
            curr_population = self.scenario_mutator.mutate(prev_population, local_pc, local_pm)
            # run
            for i in range(self.population_size):
                if self._terminal_check(start_time, self.run_hour) or self._terminal_check(local_start_time, self.local_run_hour):
                    return copy.deepcopy(local_best), False

                mutated_seed = curr_population[i]
                mutated_seed.update_id(self.seed_id)
                self.seed_id += 1
                mutated_seed: SeedWrapper = self.scenario_runner.run(mutated_seed)
                mr_oracle = self.optimal_checker.check(mutated_seed)
                fitness = self.fitness_calculator.calculate(mutated_seed)
                mutated_seed.update_oracle(oracle=mr_oracle)
                mutated_seed.update_fitness(fitness)
                mutated_seed.save()

                self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")
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

                violation_id_file = os.path.join(self.save_root, 'violation_ids.csv')
                with open(violation_id_file, 'w') as f:
                    for idx in self.nods_violations:
                        f.write(f"nods, {idx}\n")

                    for idx in self.safe_violations:
                        f.write(f"safe, {idx}\n")

                    for idx in self.mutate_violations:
                        f.write(f"mutate, {idx}\n")

                # check conditions
                if not is_safe:
                    logger.info('Find violation, exit fuzzer.')  #
                    return None, True

                curr_population[i] = mutated_seed
                if local_best is None or mutated_seed.fitness < local_best.fitness:
                    local_best = copy.deepcopy(mutated_seed)

    def run(self):
        start_time = datetime.now()
        while True:
            if self._terminal_check(start_time, self.run_hour):
                return
            self._run_global(start_time)