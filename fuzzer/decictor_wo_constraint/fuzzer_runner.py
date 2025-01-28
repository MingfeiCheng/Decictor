import copy
import os
from datetime import datetime

from loguru import logger
from omegaconf import DictConfig

from fuzzer.common.seed_wrapper import SeedWrapper
from fuzzer.decictor.core.fitness import OptimalFitness
from fuzzer.decictor_wo_constraint.core.mutator import DecictorMutator
from fuzzer.decictor.core.oracle import OptimalOracle
from fuzzer.decictor.core.selector import DecictorSelector
from tools.result_recorder import ResultRecorder
from tools.local_logger import get_instance_logger
from tools.global_config import GlobalConfig

from scenario.runner.scenario_runner import ScenarioRunner

class DecictorWoConstraint(object):
    """
    Only support one seed, load from seed folder:
    scenario_json: {seed_folder}/result/result_0.json
    [0, 1] fitness max is better
    """

    def __init__(
            self,
            cfg_fuzzer: DictConfig,
            cfg_runner: DictConfig
    ):

        # config population
        self.save_root = GlobalConfig.save_dir
        self.population_size = cfg_fuzzer.population_size  # initial corpus size
        self.run_hour = cfg_fuzzer.run_hour
        self.stop_first_nods = cfg_fuzzer.stop_first_nods

        self.seed_id = 0
        # GA
        self.curr_population = list()
        self.prev_population = list()
        # Fail case set
        self.safe_violations = list()
        self.nods_violations = list()
        self.mutate_violations = list() # record mutation error

        # result recorder
        self.result_recorder = ResultRecorder(self.save_root)

        # load refer seed - the input is the optimal seed. only one I think
        refer_seed_json = os.path.join(GlobalConfig.seed_dir, 'result/result_0.json')
        if not os.path.isfile(refer_seed_json):
            raise RuntimeError(f'{refer_seed_json} not exists')
        self.refer_seed: SeedWrapper = SeedWrapper.from_json(refer_seed_json)
        self.refer_seed.id = self.seed_id
        self.seed_id += 1

        # config runner
        self.scenario_runner = ScenarioRunner(cfg_runner)

        # config mutator
        self.scenario_mutator = DecictorMutator(cfg_fuzzer.mutator, self.refer_seed)
        # config selector
        self.scenario_selector = DecictorSelector(cfg_fuzzer.selector, cfg_fuzzer.population_size)
        # config oracle
        self.oracle_checker = OptimalOracle(self.refer_seed, cfg_fuzzer.oracle.optimal_threshold, cfg_fuzzer.oracle.grid_unit)
        # config fitness
        self.fitness_calculator = OptimalFitness(self.refer_seed, cfg_fuzzer.fitness.mode, cfg_fuzzer.fitness.line_unit)

        debug_folder = os.path.join(self.save_root, 'debug')
        log_file = os.path.join(debug_folder, f"fitness.log")
        if os.path.isfile(log_file):
            os.remove(log_file)
        self.logger_fitness = get_instance_logger(f"fitness", log_file)
        self.logger_fitness.info("Logger initialized for fitness")
        logger.info('Loaded Decictor Fuzzer')

    @staticmethod
    def _terminal_check(start_time: datetime, run_hour: float) -> bool:
        curr_time = datetime.now()
        t_delta = (curr_time - start_time).total_seconds()
        if t_delta / 3600.0 > run_hour:
            logger.info(f'Finish fuzzer as reach the time limited: {t_delta / 3600.0}/{run_hour}')
            return True
        return False

    def run(self):
        # initial curr population with refer seed
        self.curr_population = list()
        for i in range(self.population_size):
            initial_seed = copy.deepcopy(self.refer_seed)
            initial_seed.save_root = self.save_root
            initial_seed.update_fitness(0.01)
            self.curr_population.append(initial_seed)

        # start running
        start_time = datetime.now()
        find_nods = False
        while True:

            if self._terminal_check(start_time, self.run_hour):
                return

            safe_curr_population = list()
            for i in range(len(self.curr_population)):
                # 1. get source seed
                mutated_seed = copy.deepcopy(self.curr_population[i])
                # 2. mutate source seed
                mutated_seed.update_id(self.seed_id)
                self.seed_id += 1
                _, mutated_seed, mutated = self.scenario_mutator.mutate(mutated_seed)
                # 3. run mutated seed
                if mutated:
                    mutated_seed: SeedWrapper = self.scenario_runner.run(mutated_seed)
                    # 4. calculate information for both (actual we do not need this)
                    mr_oracle = self.oracle_checker.check(mutated_seed) # update record here
                    fitness = self.fitness_calculator.calculate(mutated_seed)
                    is_non_optimal = mr_oracle[1]
                    if not is_non_optimal:
                        mr_oracle_fitness = mr_oracle[2]
                        fitness += mr_oracle_fitness
                    mutated_seed.update_oracle(oracle=mr_oracle)
                    mutated_seed.update_fitness(fitness)
                    mutated_seed.save()
                    self.logger_fitness.info(f"{mutated_seed.id},{mutated_seed.fitness}")
                    # 5. update result & save violations
                    self.result_recorder.add_item(mutated_seed.record_item)
                    self.result_recorder.write_file(mutated_seed.record_columns) # write script to parse result to obtain details

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
                            find_nods = True
                        else:
                            # check mutation right
                            if is_pass_replay:
                                # not trigger rollback
                                safe_curr_population.append(mutated_seed)
                else:
                    mutated_seed.save()

                if self.stop_first_nods and find_nods:
                    logger.info('Finish fuzzer as found the first nods')
                    return

                # save existing ids
                violation_id_file = os.path.join(self.save_root, 'violation_ids.csv')
                with open(violation_id_file, 'w') as f:
                    for idx in self.nods_violations:
                        f.write(f"nods, {idx}\n")

                    for idx in self.safe_violations:
                        f.write(f"safe, {idx}\n")

                    for idx in self.mutate_violations:
                        f.write(f"mutate, {idx}\n")

            # selection
            self.prev_population = copy.deepcopy(self.curr_population)
            self.curr_population = self.scenario_selector.select(self.prev_population + safe_curr_population)