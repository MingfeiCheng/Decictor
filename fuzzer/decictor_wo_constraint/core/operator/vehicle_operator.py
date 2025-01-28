import copy
import math
import random
import numpy as np

from typing import Optional, Tuple
from omegaconf import DictConfig
from shapely.ops import unary_union

from apollo.map_parser import MapParser
from fuzzer.common.seed_wrapper import SeedWrapper
from fuzzer.decictor_wo_constraint.core.operator.region_calculator import RegionCalculator
from fuzzer.decictor_wo_constraint.core.operator.waypoint_searcher import VehicleWaypointSearcher
from scenario.config.lib.agent_type import SmallCar
from scenario.config.section.vehicle import VDAgent

class VehicleOperator(object):

    def __init__(self,
                 cfg: DictConfig,
                 region_calculator:  RegionCalculator):

        self.perturb_prob = cfg.perturb_prob
        self.max_tries = cfg.max_tries
        self.future_frames = cfg.future_frames
        self.noise_trigger = cfg.noise_trigger
        self.noise_l = cfg.noise_l
        self.noise_l_interval = cfg.noise_l_interval
        self.noise_speed = cfg.noise_speed
        self.region_calculator = region_calculator
        self._ma = MapParser.get_instance()

    def add_single_vehicle(self, source_seed: SeedWrapper, new_id: int) -> Optional[VDAgent]:

        last_waypoint = None
        waypoint_searcher = VehicleWaypointSearcher(source_seed.scenario.region, self.max_tries)
        trigger_time = random.uniform(0.0, 15.0)
        route = list()
        prohibit_region_buffer = list()

        time_list = self.region_calculator.time_list
        last_t = 0.0
        for i, t in enumerate(time_list[:-1]):
            motion_prohibit_region_lst = list()
            for f_i in range(self.future_frames): # delta time is future_frames * time_interval in time_lst
                future_index = i + f_i if (i + f_i) < len(time_list) - 1 else len(time_list) - 2
                future_next_index = future_index + 1
                motion_prohibit_region_lst.append(self.region_calculator.get_frame_region(
                    time_list[future_index], time_list[future_next_index]
                ))
            motion_prohibit_region = unary_union(motion_prohibit_region_lst)
            prohibit_region_buffer.append(motion_prohibit_region)

            delta_t = t - last_t
            last_t = t

            if t <= trigger_time:
                continue

            prohibit_region = unary_union(prohibit_region_buffer)
            next_waypoint = waypoint_searcher.search(last_waypoint, delta_t, prohibit_region, SmallCar())
            if next_waypoint is None:
                return None
            route.append(copy.deepcopy(next_waypoint))
            last_waypoint = next_waypoint

        # create vehicle object
        route[-1].origin_speed = 0.0
        vehicle_id = new_id
        vehicle_agent = VDAgent(vehicle_id, mutable=True, route=route, agent_type=SmallCar(), origin_trigger=trigger_time, noise_trigger=0.0)
        return vehicle_agent

    def remove_vehicles(self, seed: SeedWrapper, num: int) -> Tuple[SeedWrapper, int]:
        scenario = seed.scenario
        vehicle_ids = scenario.vehicles.mutant_ids
        if len(vehicle_ids) <= 0:
            remove_count = 0
        else:
            # remove rate
            remove_count = 0
            # total_number = len(vehicle_ids)
            for _ in range(num):
                if len(vehicle_ids) <= 0:
                    break
                target_id = random.choice(vehicle_ids)
                res = scenario.vehicles.remove_agent(target_id)
                if res:
                    remove_count += 1
            seed.scenario = scenario
        return seed, remove_count

    def perturb_vehicles(self, source_seed: SeedWrapper) -> Tuple[SeedWrapper, int]:
        scenario = source_seed.scenario
        vehicle_ids = scenario.vehicles.mutant_ids
        if len(vehicle_ids) <= 0:
            perturb_count = 0
        else:
            # perturb agent
            perturb_count = 0
            for i in range(len(vehicle_ids)):
                target_agent = scenario.vehicles.get_agent(vehicle_ids[i])
                if random.random() < self.perturb_prob:
                    perturbed_agent, is_perturb = self.perturb_single_vehicle(target_agent)
                    # TODO: check this agent
                    if is_perturb:
                        scenario.vehicles.update_agent(vehicle_ids[i], perturbed_agent)
                        perturb_count += 1
        source_seed.scenario = scenario
        return source_seed, perturb_count

    def perturb_single_vehicle(self, agent: VDAgent) -> Tuple[VDAgent, bool]:
        # TODO: add region checker
        noise_trigger_range = self.noise_trigger
        noise_lateral_range = self.noise_l
        noise_lateral_interval_range = self.noise_l_interval
        noise_speed_range = self.noise_speed

        perturb_agent = copy.deepcopy(agent)

        # 1. change trigger time - accumulate error
        noise_trigger = random.uniform(noise_trigger_range[0], noise_trigger_range[1])
        perturb_trigger = float(np.clip(perturb_agent.origin_trigger + noise_trigger, 0.0, None))
        perturb_agent.noise_trigger = perturb_trigger - perturb_agent.origin_trigger

        # 2. change
        perturb_route = copy.deepcopy(perturb_agent.route)
        perturb_route_length = len(perturb_route)

        # 2-1. change l
        noise_lateral_interval = random.randint(noise_lateral_interval_range[0], noise_lateral_interval_range[1])
        noise_lateral = 0
        noise_direction = 1

        for i in range(perturb_route_length):
            if i % noise_lateral_interval == 0:
                noise_lateral = random.uniform(noise_lateral_range[0], noise_lateral_range[1])
                noise_direction = random.choice([-1, 1])

            l_max = 1.0 # min(perturb_route[i].left_width, perturb_route[i].right_width)

            source_wp_ori_l = perturb_route[i].origin.l
            perturb_l = np.clip(perturb_route[i].l + noise_direction * noise_lateral, -l_max * 0.2, l_max * 0.2)
            perturb_route[i].perturb.l = perturb_l - source_wp_ori_l

            perturb_x, perturb_y, _ = self._ma.get_coordinate(perturb_route[i].lane_id, perturb_route[i].s, perturb_route[i].l)
            perturb_route[i].perturb.x = perturb_x - perturb_route[i].origin.x
            perturb_route[i].perturb.y = perturb_y - perturb_route[i].origin.y

        ## change heading
        for i in range(perturb_route_length):
            if i == perturb_route_length - 1:
                perturb_route[i].origin.heading = perturb_route[i - 1].origin.heading
            else:
                x1, y1 = perturb_route[i].x, perturb_route[i].y
                x2, y2 = perturb_route[i + 1].x, perturb_route[i + 1].y
                perturb_route[i].origin.heading = math.atan2(y2 - y1, x2 - x1)

        # 2-2. change speed
        for i in range(perturb_route_length):
            noise_speed = random.uniform(noise_speed_range[0], noise_speed_range[1])
            speed = np.clip(perturb_route[i].speed + noise_speed, 0.3, 25.0)
            perturb_route[i].perturb_speed = speed - perturb_route[i].origin_speed

        perturb_agent.route = perturb_route
        return perturb_agent, True
