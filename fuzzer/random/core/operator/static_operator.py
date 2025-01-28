import copy
import random

from typing import TypeVar, Optional, Tuple
from omegaconf import DictConfig
from shapely.geometry import Point, Polygon

from apollo.map_parser import MapParser
from fuzzer.common.seed_wrapper import SeedWrapper
from scenario.config.common import LaneBoundaryWaypoint, PositionUnit
from scenario.config.lib.agent_type import AgentType, TrafficCone
from scenario.config.section.static import STAgent
from tools.utils import generate_polygon

from fuzzer.random.core.operator.region_calculator import RegionCalculator
from shapely.ops import unary_union

AgentTypeClass = TypeVar("AgentTypeClass", bound=AgentType)

class StaticOperator(object):

    def __init__(self,
                 cfg: DictConfig,
                 region_calculator:  RegionCalculator):

        self.perturb_prob = cfg.perturb_prob
        self.max_tries = cfg.max_tries
        self.noise_x = [-0.5, 0.5]
        self.noise_y = [-0.5, 0.5]
        self.region_calculator = region_calculator
        self._ma = MapParser.get_instance()

    def add_single_static(self, source_seed: SeedWrapper, new_id: int) -> Optional[STAgent]:
        """
        Random add statics:
        1. same lane - keep distance at least 4.0
        """
        prohibit_region = self.region_calculator.get_global_region()

        # 2. Add prohibited region of new added statics
        prohibit_region_lst = [prohibit_region]
        for st in source_seed.scenario.statics.agents:
            prohibit_region_lst.append(st.get_initial_polygon())
        prohibit_region = unary_union(prohibit_region_lst)

        # some patterns and calculate the conflict based on the prohibit region
        lanes_static = copy.deepcopy(source_seed.scenario.region.lanes_static)
        random.shuffle(lanes_static)
        waypoint = None
        for i, lane_id in enumerate(lanes_static):
            lane_left_line, lane_right_line = self._ma.get_lane_boundary_curve(lane_id)
            side = random.choice(['right', 'left'])
            if side == 'right':
                lane_line = lane_right_line
            else:
                lane_line = lane_left_line

            lane_line_length = lane_line.length
            num_points = int(lane_line_length / 1.0) + 1
            num_points = max(num_points, 2)
            sample_points = [lane_line.interpolate(i / float(num_points - 1), normalized=True) for i in range(num_points)]
            points_outside_polygon = [point for point in sample_points if not prohibit_region.contains(point)]
            if len(points_outside_polygon) <= 0:
                continue

            # get waypoint
            random.shuffle(points_outside_polygon)
            for j in range(self.max_tries):
                if j >= len(points_outside_polygon):
                    break
                point = points_outside_polygon[j]
                point_s = lane_line.project(point)
                point_location, point_heading = self._ma.get_coordinate_and_heading_any(lane_line, point_s)

                lane_center_line = self._ma.get_lane_central_curve(lane_id)
                s = lane_center_line.project(point)
                # create waypoint
                waypoint = LaneBoundaryWaypoint(
                    origin=PositionUnit(
                        lane_id=lane_id,
                        s=s,
                        l=0.0,
                        x=point_location.x,
                        y=point_location.y,
                        z=0.0,
                        heading=point_heading
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
                    boundary=side,
                )

            if waypoint is None:
                continue
            else:
                break

        if waypoint is None:
            return None
        else:
            static_id = new_id
            static_agent = STAgent(static_id, mutable=True, route=[waypoint], agent_type=TrafficCone(),
                                   origin_trigger=0.0, noise_trigger=0.0)
        return static_agent

    def remove_statics(self, seed: SeedWrapper, num: int) -> Tuple[SeedWrapper, int]:
        scenario = seed.scenario

        static_ids = scenario.statics.mutant_ids
        if len(static_ids) <= 0:
            remove_count = 0
        else:
            # remove rate
            remove_count = 0
            for _ in range(num):
                if len(static_ids) <= 0:
                    break
                target_id = random.choice(static_ids)
                res = scenario.statics.remove_agent(target_id)
                if res:
                    remove_count += 1
            seed.scenario = scenario
        return seed, remove_count

    def perturb_statics(self, source_seed: SeedWrapper) -> Tuple[SeedWrapper, int]:
        scenario = source_seed.scenario
        static_ids = scenario.statics.mutant_ids
        if len(static_ids) <= 0:
            perturb_count = 0
        else:
            perturb_count = 0
            for i in range(len(static_ids)):
                target_agent = scenario.statics.get_agent(static_ids[i])
                if random.random() < self.perturb_prob:
                    prohibited_region = self.region_calculator.get_global_region_except(target_agent.id)
                    prohibit_region_lst = [prohibited_region]
                    for st in source_seed.scenario.statics.agents:
                        if st.id == target_agent.id:
                            continue
                        prohibit_region_lst.append(st.get_initial_polygon())
                    prohibited_region = unary_union(prohibit_region_lst)
                    perturbed_agent, is_perturb = self._perturb_single_static(target_agent, prohibited_region)
                    if is_perturb:
                        scenario.statics.update_agent(static_ids[i], perturbed_agent)
                        perturb_count += 1
                        pass
                source_seed.scenario = scenario
        return source_seed, perturb_count

    def _perturb_single_static(self,
                               agent: STAgent, prohibited_region: Polygon) -> Tuple[STAgent, bool]:

        noise_x_range = self.noise_x # [-0.05, 0.05]  # x range for perturbation
        noise_y_range = self.noise_y #[-0.05, 0.05]  # y range for perturbation
        perturb_agent = copy.deepcopy(agent)
        is_perturb = False
        # 1. chane lane side
        if random.random() < self.perturb_prob / 2.0:
            agent_waypoint: LaneBoundaryWaypoint = perturb_agent.route[0]
            origin_side = agent_waypoint.boundary
            lane_left_line, lane_right_line = self._ma.get_lane_boundary_curve(agent_waypoint.lane_id)
            lane_center_line = self._ma.get_lane_central_curve(agent_waypoint.lane_id)

            if origin_side == 'left':
                target_side = random.choice(['right', 'center'])
                if target_side == 'right':
                    target_line = lane_right_line
                else:
                    target_line = lane_center_line
            elif origin_side == 'right':
                target_side = random.choice(['left', 'center'])
                if target_side == 'left':
                    target_line = lane_left_line
                else:
                    target_line = lane_center_line
            else:
                target_side = random.choice(['left', 'right'])
                if target_side == 'left':
                    target_line = lane_left_line
                else:
                    target_line = lane_right_line

            center_s = agent_waypoint.s
            center_point, heading = self._ma.get_coordinate_and_heading(agent_waypoint.lane_id, center_s)
            center_point = Point([center_point.x, center_point.y])
            point_s = target_line.project(center_point)
            point_location, point_heading = self._ma.get_coordinate_and_heading_any(target_line, point_s)
            point_shape = perturb_agent.agent_type
            point_polygon, _ = generate_polygon(
                point_location.x,
                point_location.y,
                point_heading,
                front_l=point_shape.length / 2.0,
                back_l=point_shape.length / 2.0,
                width=point_shape.width
            )
            if point_polygon.intersects(prohibited_region):
                pass
            else:
                agent_waypoint.boundary = target_side
                agent_waypoint.origin.s = center_s
                agent_waypoint.origin.x = point_location.x
                agent_waypoint.origin.y = point_location.y
                agent_waypoint.origin.heading = point_heading
                perturb_agent.route[0] = agent_waypoint
                is_perturb = True

        # 2. add noises
        point_shape = perturb_agent.agent_type
        agent_waypoint: LaneBoundaryWaypoint = perturb_agent.route[0]
        tries = 0
        while tries < self.max_tries:
            tries += 1
            noise_x = random.uniform(noise_x_range[0], noise_x_range[1])
            noise_y = random.uniform(noise_y_range[0], noise_y_range[1])
            point_polygon, _ = generate_polygon(
                agent_waypoint.origin.x + noise_x,
                agent_waypoint.origin.y + noise_y,
                agent_waypoint.heading,
                front_l=point_shape.length / 2.0,
                back_l=point_shape.length / 2.0,
                width=point_shape.width
            )
            if point_polygon.intersects(prohibited_region):
                continue
            else:
                agent_waypoint.perturb.x = noise_x
                agent_waypoint.perturb.y = noise_y
                perturb_agent.route[0] = agent_waypoint
                is_perturb = True
                break

        return perturb_agent, is_perturb

