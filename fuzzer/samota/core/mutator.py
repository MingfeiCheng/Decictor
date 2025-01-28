from __future__ import annotations

import random
import copy
import math
import numpy as np

from loguru import logger
from datetime import datetime
from shapely.geometry import Polygon
from typing import TypeVar, List, Optional

from apollo.map_parser import MapParser
from scenario.config.section.vehicle import VDAgent
from scenario.config.common import MotionWaypoint, PositionUnit
from scenario.config.lib.agent_type import SmallCar, Bicycle
from fuzzer.common.seed_wrapper import SeedWrapper

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

    def __init__(self):
        self._ma = MapParser.get_instance()
        self.s_interval = 2.0

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

    def _get_new_vehicle(self, seed: SeedWrapperClass, start_lane_id: str, speed: float, trigger_time: float) -> Optional[VDAgent]:
        # select vehicle lanes
        new_id = seed.scenario.vehicles.get_new_id()
        start_lane = start_lane_id
        vehicle_route_lanes = self._ma.get_random_route_from_start_lane(start_lane, 10)

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
        lane_speed = speed
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
            lane_speed = speed
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
        vd_trigger = trigger_time
        agent = VDAgent(new_id, mutable=True, route=route, agent_type=SmallCar(), origin_trigger=vd_trigger, noise_trigger=0.0)
        return agent

    def _get_new_bicycle(self, seed: SeedWrapperClass, start_lane_id: str, speed: float, trigger_time: float) -> Optional[VDAgent]:
        # select vehicle lanes
        new_id = seed.scenario.vehicles.get_new_id()
        start_lane = start_lane_id
        vehicle_route_lanes = self._ma.get_random_route_from_start_lane(start_lane, 10)

        start_s_lst = self._ma.get_waypoint_s_for_lane(start_lane, self.s_interval)
        start_s_index = None
        start_s_indexes = list(np.arange(0, len(start_s_lst) - 1))
        random.shuffle(start_s_indexes)
        for item in start_s_indexes:
            start_s = start_s_lst[item]
            point, heading = self._ma.get_coordinate_and_heading(start_lane, start_s)
            agent_type = Bicycle()
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
        lane_speed = speed
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
            lane_speed = speed
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
        vd_trigger = trigger_time
        agent = VDAgent(new_id, mutable=True, route=route, agent_type=Bicycle(), origin_trigger=vd_trigger, noise_trigger=0.0)
        return agent

    def get_scenario(self, refer_seed: SeedWrapperClass, candidate_value: List):
        """
        workflow:
            1. add static obstacles before add vehicles
            2. if static obstacles are enough -> add vehicles
        specific flow:
            1. ADD -> check total number
            2. If > total number limitation -> remove
        """
        # 0 Scenario ID - skip
        # 1 Vehicle_in_front
        # 2 vehicle_in_adjcent_lane
        # 3 vehicle_in_opposite_lane
        # 4 vehicle_in_front_two_wheeled
        # 5 vehicle_in_adjacent_two_wheeled
        # 6 vehicle_in_opposite_two_wheeled
        # 7 Target Speed
        # 8 Trigger Time
        m_start_time = datetime.now()
        logger.debug(candidate_value)
        ma = MapParser.get_instance()
        mutated_seed = copy.deepcopy(refer_seed)

        lane_ego_start = mutated_seed.scenario.egos.agents[0].route[0].lane_id
        pending_vehicle_lanes = []

        # 1 Vehicle_in_front
        if candidate_value[1] == 0:
            pass
        elif candidate_value[1] == 1:
            pending_lanes = ma.get_successor_lanes(lane_ego_start)
            pending_lanes += [lane_ego_start]
            if len(pending_lanes) > 0:
                pending_vehicle_lanes.append(random.choice(pending_lanes))

        # 2 vehicle_in_adjcent_lane
        if candidate_value[2] == 0:
            pass
        elif candidate_value[2] == 1:
            select_lane = ma.get_neighbors(lane_ego_start, direct='forward', side='both')
            if len(select_lane) > 0:
                pending_vehicle_lanes.append(random.choice(select_lane))

        # 3 vehicle_in_opposite_lane
        if candidate_value[3] == 0:
            pass
        elif candidate_value[3] == 1:
            select_lane = ma.get_neighbors(lane_ego_start, direct='reverse', side='both')
            if len(select_lane) > 0:
                pending_vehicle_lanes.append(random.choice(select_lane))

        pending_bicycle_lanes = []
        # 4 vehicle_in_front_two_wheeled
        if candidate_value[4] == 0:
            pass
        elif candidate_value[4] == 1:
            pending_lanes = ma.get_successor_lanes(lane_ego_start)
            pending_lanes += [lane_ego_start]
            if len(pending_lanes) > 0:
                pending_bicycle_lanes.append(random.choice(pending_lanes))

        # 5 vehicle_in_adjacent_two_wheeled
        if candidate_value[5] == 0:
            pass
        elif candidate_value[5] == 1:
            select_lane = ma.get_neighbors(lane_ego_start, direct='forward', side='both')
            if len(select_lane) > 0:
                pending_bicycle_lanes.append(random.choice(select_lane))

        # 6 vehicle_in_opposite_two_wheeled
        if candidate_value[6] == 0:
            pass
        elif candidate_value[6] == 1:
            select_lane = ma.get_neighbors(lane_ego_start, direct='reverse', side='both')
            if len(select_lane) > 0:
                pending_bicycle_lanes.append(random.choice(select_lane))

        speed = candidate_value[7]
        trigger_time = candidate_value[8]
        # generate participants
        # vehicle
        for v_lane_id in pending_vehicle_lanes:
            new_agent = self._get_new_vehicle(mutated_seed, v_lane_id, speed, trigger_time)
            if new_agent is not None:
                mutated_seed.scenario.vehicles.add_agent(copy.deepcopy(new_agent))

        for b_lane_id in pending_bicycle_lanes:
            new_agent = self._get_new_bicycle(mutated_seed, b_lane_id, speed, trigger_time)
            if new_agent is not None:
                mutated_seed.scenario.vehicles.add_agent(copy.deepcopy(new_agent))

        m_end_time = datetime.now()
        logger.error('Mutation Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())
        return mutated_seed