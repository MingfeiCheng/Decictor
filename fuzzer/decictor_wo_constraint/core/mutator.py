import os
import copy
import random

from datetime import datetime
from typing import Tuple
from omegaconf import DictConfig

from fuzzer.common.seed_wrapper import SeedWrapper
from fuzzer.decictor_wo_constraint.core.operator.region_calculator import RegionCalculator
from fuzzer.decictor_wo_constraint.core.operator.static_operator import StaticOperator
from fuzzer.decictor_wo_constraint.core.operator.vehicle_operator import VehicleOperator
from tools.global_config import GlobalConfig
from tools.local_logger import get_instance_logger

class DecictorMutator:

    def __init__(self, cfg: DictConfig, adv_seed: SeedWrapper):

        gc = GlobalConfig.get_instance()
        # print(cfg)
        self.prob_type = cfg.prob_type # static <= this value
        self.prob_perturb = cfg.prob_perturb
        self.prob_remove = cfg.prob_remove
        self.motion_interval = cfg.motion_interval
        self.limit_static = cfg.limit_static
        self.limit_vehicle = cfg.limit_vehicle
        self.min_add_static = cfg.min_add_static
        self.max_add_static = cfg.max_add_static
        self.min_remove_static = cfg.min_remove_static
        self.max_remove_static = cfg.max_remove_static
        self.min_remove_vehicle = cfg.min_remove_vehicle
        self.max_remove_vehicle = cfg.max_remove_vehicle
        self.max_tries = cfg.max_tries

        self.operate_type = 'static' # ['static', 'vehicle']
        self.region_calculator_static = RegionCalculator(cfg.static_mutator, adv_seed)
        self.static_operator = StaticOperator(cfg.static_mutator, self.region_calculator_static)
        self.region_calculator_vehicle = RegionCalculator(cfg.vehicle_mutator, adv_seed)
        self.vehicle_operator = VehicleOperator(cfg.vehicle_mutator, self.region_calculator_vehicle)

        save_root = gc.cfg.save_root
        save_folder = os.path.join(save_root, 'debug')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        log_file = os.path.join(save_folder, f"mutator.log")
        if os.path.isfile(log_file):
            os.remove(log_file)

        self.logger = get_instance_logger(f"mutator", log_file)
        self.logger.info("Logger initialized for mutator")

    def _add_operator(self, next_seed: SeedWrapper) -> Tuple[SeedWrapper, bool]:
        mutated = False
        # Step 1. Add Operators
        if self.operate_type == 'static':
            # 1-1 Add statics
            target_count = int(random.uniform(self.min_add_static, self.max_add_static + 1.0))
            actual_count = 0
            for _ in range(target_count):
                agent_id = next_seed.scenario.statics.get_new_id()
                tries = 0
                while tries < self.max_tries:
                    tries += 1
                    agent = self.static_operator.add_single_static(next_seed, new_id=agent_id)
                    if agent is None:
                        continue
                    next_seed.scenario.statics.add_agent(agent)
                    actual_count += 1
                    break
            # update flag
            if actual_count > 0:
                mutated = True
            self.logger.info(f"Add Static: {actual_count}/{target_count}")
        else:
            # 1-2 Add single vehicles
            target_count = 1
            actual_count = 0
            for _ in range(target_count):
                agent_id = next_seed.scenario.vehicles.get_new_id()
                tries = 0
                while tries < self.max_tries:
                    tries += 1
                    agent = self.vehicle_operator.add_single_vehicle(next_seed, new_id=agent_id)
                    if agent is None:
                        continue
                    next_seed.scenario.vehicles.add_agent(agent)
                    actual_count += 1
                    break
            # update flag
            if actual_count > 0:
                mutated = True
            self.logger.info(f"Add Vehicle: {actual_count}/{target_count}")
        return next_seed, mutated

    def _perturb_operator(self, next_seed: SeedWrapper) -> Tuple[SeedWrapper, bool]:
        mutated = False
        if self.operate_type == 'static':
            # 2-1 Perturb statics
            next_seed, perturb_num = self.static_operator.perturb_statics(next_seed)
            # update flag
            if perturb_num > 0:
                mutated = True
            self.logger.info(f"Perturb Static: {perturb_num}")
        else:
            # 2-2 Perturb vehicles
            next_seed, perturb_num = self.vehicle_operator.perturb_vehicles(next_seed)
            # update flag
            if perturb_num > 0:
                mutated = True
            self.logger.info(f"Perturb Vehicle: {perturb_num}")
        return next_seed, mutated

    def _remove_operator(self, next_seed: SeedWrapper, added_flag: bool) -> Tuple[SeedWrapper, bool]:
        # Extend add operator
        mutated = False
        if self.operate_type == 'static':
            # 3-1 remove statics
            existing_static_num = len(next_seed.scenario.statics.agents)
            if existing_static_num > self.limit_static * 1.1 or not added_flag:
                target_count = max(int(existing_static_num * 0.2), 1) #int(random.uniform(self.min_remove_static, self.max_remove_static))
            else:
                target_count = 0
            next_seed, actual_count = self.static_operator.remove_statics(next_seed, target_count)
            if actual_count > 0:
                mutated = True
            self.logger.info(f"Remove Static: {actual_count}/{target_count}")
        else:
            # 3-2 remove vehicles
            existing_vehicle_num = len(next_seed.scenario.vehicles.agents)
            if existing_vehicle_num > self.limit_vehicle * 1.1 or not added_flag:
                target_count = max(int(existing_vehicle_num * 0.2), 1) # int(random.uniform(self.min_remove_vehicle, self.max_remove_vehicle))
            else:
                target_count = 0
            next_seed, actual_count = self.vehicle_operator.remove_vehicles(next_seed, target_count)
            if actual_count > 0:
                mutated = True
            self.logger.info(f"Remove Vehicle: {actual_count}/{target_count}")
        return next_seed, mutated

    def _remove_operator_pre(self, next_seed: SeedWrapper) -> Tuple[SeedWrapper, bool]:
        mutated = False
        if self.operate_type == 'static':
            # 3-1 remove statics
            existing_static_num = len(next_seed.scenario.statics.agents)
            if existing_static_num > self.limit_static * 1.1:
                target_count = int(random.uniform(self.min_remove_static, self.max_remove_static)) #random.choice([0, 1, 2])
            else:
                target_count = 0
            next_seed, actual_count = self.static_operator.remove_statics(next_seed, target_count)
            if actual_count > 0:
                mutated = True
            self.logger.info(f"Pre Remove Static: {actual_count}/{target_count}")
        else:
            # 3-2 remove vehicles
            existing_vehicle_num = len(next_seed.scenario.vehicles.agents)
            if existing_vehicle_num > self.limit_vehicle * 1.1:
                target_count = random.choice([1, 2])
            else:
                target_count = 0
            next_seed, actual_count = self.vehicle_operator.remove_vehicles(next_seed, target_count)
            if actual_count > 0:
                mutated = True
            self.logger.info(f"Pre Remove Vehicle: {actual_count}/{target_count}")
        return next_seed, mutated

    def mutate(self, source_seed: SeedWrapper) -> Tuple[SeedWrapper, SeedWrapper, bool]:
        self.logger.info(f"========== Mutate Seed {source_seed.id} ==========")
        mutate_start_time = datetime.now()
        next_seed = copy.deepcopy(source_seed)

        if random.random() < self.prob_type:
            self.operate_type = 'static'
        else:
            self.operate_type = 'vehicle'

        # Step 0. Random remove minor samples
        # mutated = False
        next_seed, mutated = self._remove_operator_pre(next_seed)

        # Step 1. Calculate the time list & prohibit region for adding and perturbation
        self.region_calculator_static.preprocess(next_seed, self.motion_interval)
        self.region_calculator_vehicle.preprocess(next_seed, self.motion_interval)

        # Step 2. Add & Remove Operators
        add_flag = True
        next_seed, added = self._add_operator(next_seed)
        if added:
            mutated = True
        else:
            add_flag = False

        # Step 3. Remove
        if random.random() < self.prob_remove:
            next_seed, removed = self._remove_operator(next_seed, add_flag)
            if removed:
                mutated = True

        # Step 4: perturbation with small noises
        if random.random() < self.prob_perturb:
            next_seed, perturbed = self._perturb_operator(next_seed)
            if perturbed:
                mutated = True

        self.logger.info(f"Total: Static ({len(next_seed.scenario.statics.agents)}) Vehicle ({len(next_seed.scenario.vehicles.agents)})")
        self.logger.info(f"========== Mutate End {(datetime.now() - mutate_start_time).total_seconds()} ==========")
        return source_seed, next_seed, mutated