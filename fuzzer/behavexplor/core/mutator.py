from __future__ import annotations

import copy
import math
import random
from datetime import datetime
from typing import Tuple, TypeVar, List, Optional

import numpy as np
from loguru import logger
from omegaconf import DictConfig
from shapely.geometry import Polygon, LineString

from apollo.map_parser import MapParser
from fuzzer.common.seed_wrapper import SeedWrapper
from scenario.config.common import MotionWaypoint, PositionUnit
from scenario.config.lib.agent_type import SmallCar
from scenario.config.section.vehicle import VDAgent

SeedWrapperClass = TypeVar("SeedWrapperClass", bound=SeedWrapper)

MAX_SPEED = 25.0
MIN_SPEED = 0.3

def generate_polygon(position: List, shape: List) -> Polygon:
    """
    Generate polygon for a perception obstacle
    position : [x, y, z, heading]
    shape: [length, width, height]
    :returns:
        List with 4 Point3D objects representing the polygon of the obstacle
    :rtype: List[Point3D]
    """
    points = []
    theta = position[3]
    length = shape[0]
    width = shape[1]
    half_l = length / 2.0
    half_w = width / 2.0
    sin_h = math.sin(theta)
    cos_h = math.cos(theta)
    vectors = [(half_l * cos_h - half_w * sin_h,
                half_l * sin_h + half_w * cos_h),
               (-half_l * cos_h - half_w * sin_h,
                - half_l * sin_h + half_w * cos_h),
               (-half_l * cos_h + half_w * sin_h,
                - half_l * sin_h - half_w * cos_h),
               (half_l * cos_h + half_w * sin_h,
                half_l * sin_h - half_w * cos_h)]
    for x, y in vectors:
        p_x = position[0] + x
        p_y = position[1] + y
        points.append([p_x, p_y])

    return Polygon(points)

class ScenarioMutator:

    def __init__(self, cfg: DictConfig):
        self._ma = MapParser.get_instance()
        self.s_interval = cfg.s_interval
        self.npc_number = int(cfg.npc_number * 1.5)

    ########## Basic checkers ##########
    def _is_conflict(self, seed: SeedWrapperClass, agent_polygon: Polygon) -> bool:
        # only check initial localization
        ego_polygon = seed.scenario.egos.agents[0].get_initial_polygon()
        if agent_polygon.distance(ego_polygon) < 5:
            return True

        # other vds
        for exist_agent in seed.scenario.vehicles.agents:
            exist_agent_polygon = exist_agent.get_initial_polygon()
            if agent_polygon.distance(exist_agent_polygon) < 1.5:
                return True

        return False

    def _get_new_vehicle(self, seed: SeedWrapperClass, require_intersection: bool = True) -> Optional[VDAgent]:

        new_id = seed.scenario.vehicles.get_new_id()
        lanes_vehicle = seed.scenario.region.lanes_vehicle + seed.scenario.region.lanes_ego
        lanes_vehicle = list(set(lanes_vehicle))

        start_lane = random.choice(lanes_vehicle)
        vehicle_route_lanes = self._ma.get_random_route_from_start_lane(start_lane, 10)

        has_intersection = False
        ego_trace = LineString(seed.record.ego.trace_pts)
        for vr_lane_id in vehicle_route_lanes:
            vr_lane = self._ma.get_lane_polygon(vr_lane_id)
            if ego_trace.intersects(vr_lane):
                has_intersection = True
                break

        if not has_intersection and require_intersection:
            return None

        start_s_lst = self._ma.get_waypoint_s_for_lane(start_lane, self.s_interval)
        start_s_index = None
        start_s_indexes = list(np.arange(0, len(start_s_lst) - 1))
        random.shuffle(start_s_indexes)
        for item in start_s_indexes:
            start_s = start_s_lst[item]
            point, heading = self._ma.get_coordinate_and_heading(start_lane, start_s)
            agent_type = SmallCar()
            agent_polygon = generate_polygon([point.x, point.y, 0.0, heading],
                                             [agent_type.length, agent_type.width, agent_type.height])
            if self._is_conflict(seed, agent_polygon):
                continue
            else:
                start_s_index = item
                break

        if start_s_index is None:
            return None

        start_s_lst = start_s_lst[start_s_index: ]
        route = list()
        # add start point
        lane_speed = random.uniform(MIN_SPEED, MAX_SPEED)
        for i, s in enumerate(start_s_lst):
            if i == 0:
                waypoint_speed = 0.0
            else:
                waypoint_speed = lane_speed
            lane_id = start_lane
            point, heading = self._ma.get_coordinate_and_heading(lane_id, s)
            route.append(MotionWaypoint(
                origin=PositionUnit(
                    lane_id=lane_id,
                    s=s,
                    l=0.0,
                    x=point.x,
                    y=point.y,
                    z=0.0,
                    heading=heading
                ),
                perturb=PositionUnit(
                    None,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ),
                origin_speed=waypoint_speed,
                perturb_speed=0.0,
                is_junction=self._ma.is_junction_lane(lane_id)
            ))

        # for mid
        for lane_index, lane_id in enumerate(vehicle_route_lanes[1:]):
            lane_s_lst = self._ma.get_waypoint_s_for_lane(lane_id, self.s_interval)
            lane_speed = random.uniform(MIN_SPEED, MAX_SPEED)
            for s_index, s in enumerate(lane_s_lst):
                waypoint_speed = lane_speed
                point, heading = self._ma.get_coordinate_and_heading(lane_id, s)
                route.append(MotionWaypoint(
                    origin=PositionUnit(
                        lane_id=lane_id,
                        s=s,
                        l=0.0,
                        x=point.x,
                        y=point.y,
                        z=0.0,
                        heading=heading
                    ),
                    perturb=PositionUnit(
                        None,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ),
                    origin_speed=waypoint_speed,
                    perturb_speed=0.0,
                    is_junction=self._ma.is_junction_lane(lane_id)
                ))

        route[-1].origin_speed = 0.0
        vd_trigger = random.uniform(0.0, 15.0)
        agent = VDAgent(new_id, mutable=True, route=route, agent_type=SmallCar(), origin_trigger=vd_trigger, noise_trigger=0.0)
        return agent

    def _small_mutation(self, seed: SeedWrapperClass) -> SeedWrapperClass:
        vds: List[VDAgent] = seed.scenario.vehicles.agents
        for i in range(len(vds)):
            if random.random() > 0.5:
                continue
            vd_i = vds[i]
            # trigger time
            vd_trigger = float(np.clip(random.gauss(vd_i.trigger, 2.0), 0.0, 15.0))
            noise_trigger = vd_trigger - vd_i.origin_trigger
            vd_i.noise_trigger = noise_trigger

            # speed
            route_length = len(vd_i.route)
            mutate_speed_interval = 10
            # mutated_indexes = random.sample(range(len(vd_i.route)), min(len(vd_i.route), 5))

            for m_id_ in range(int(route_length / mutate_speed_interval)):
                m_id = m_id_ * 10
                m_id = min(m_id, route_length - 1)

                # for m_id in mutated_indexes:
                prev_speed = vd_i.route[m_id].speed
                mutated_speed = float(np.clip(random.gauss(prev_speed, 5), MIN_SPEED, MAX_SPEED))
                perturb_speed = mutated_speed - vd_i.route[m_id].origin_speed
                vd_i.route[m_id].perturb_speed = perturb_speed
            vds[i] = vd_i
        seed.scenario.vehicles.agents = vds
        return seed

    def _large_mutation(self, seed: SeedWrapperClass) -> SeedWrapperClass:
        vds = seed.scenario.vehicles.agents
        vd_ids = seed.scenario.vehicles.ids
        for i in range(len(vds)):
            if random.random() > 0.5:
                target_id = vd_ids[i]
                _ = seed.scenario.vehicles.remove_agent(target_id)
                single_tries = 0
                while True:
                    if single_tries < 50 and random.random() < 0.8:
                        new_agent = self._get_new_vehicle(seed, require_intersection=True)
                    else:
                        new_agent = self._get_new_vehicle(seed, require_intersection=False)

                    if new_agent is not None:
                        seed.scenario.vehicles.add_agent(copy.deepcopy(new_agent))
                        break
                    else:
                        single_tries += 1
        return seed

    def mutate(self, source_seed: SeedWrapperClass, mutation_type: str) -> Tuple[SeedWrapperClass, SeedWrapperClass]:
        mutated_seed = copy.deepcopy(source_seed)
        m_start_time = datetime.now()
        if mutation_type == 'small':
            mutated_seed = self._small_mutation(mutated_seed)
        else:
            mutated_seed = self._large_mutation(mutated_seed)
        m_end_time = datetime.now()
        logger.error('Mutation Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())
        return mutated_seed, source_seed

    def get_random_traffic(self, source_seed: SeedWrapperClass):
        mutated_seed = copy.deepcopy(source_seed)

        single_tries = 0
        while len(mutated_seed.scenario.vehicles.agents) < self.npc_number:
            if single_tries < 50 and random.random() < 0.85:
                new_agent = self._get_new_vehicle(mutated_seed, require_intersection=True)
            else:
                new_agent = self._get_new_vehicle(mutated_seed, require_intersection=False)

            if new_agent is not None:
                mutated_seed.scenario.vehicles.add_agent(copy.deepcopy(new_agent))
                single_tries = 0
            else:
                single_tries +=1

        return mutated_seed