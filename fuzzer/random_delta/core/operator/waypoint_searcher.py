import copy
import random
from typing import TypeVar, Optional

import numpy as np
from shapely.geometry import Polygon, LineString, Point

from apollo.map_parser import MapParser
from scenario.config.common import MotionWaypoint, PositionUnit, WalkerMotionWaypoint
from scenario.config.lib.agent_type import AgentType
from scenario.config.section.region import RESection
from tools.utils import generate_polygon

AgentTypeClass = TypeVar("AgentTypeClass", bound=AgentType)

class VehicleWaypointSearcher:
    """
    Search NPC vehicle route according to:
    1. lanes vehicle
    2. prohibited region
    """

    def __init__(self, region: RESection, max_tries: int):

        self.lanes = region.lanes_vehicle
        self._ma = MapParser.get_instance()
        self.max_tries = max_tries

    def _sample_start_waypoint(self,
                               prohibit_region: Polygon,
                               agent_type: AgentTypeClass) -> Optional[MotionWaypoint]:
        lanes = copy.deepcopy(self.lanes)
        random.shuffle(lanes)
        for i, lane_id in enumerate(lanes):
            lane_line = self._ma.get_lane_central_curve(lane_id)
            lane_line_length = lane_line.length
            num_points = int(lane_line_length / 1.0) + 1
            num_points = max(num_points, 2)
            sample_points = [lane_line.interpolate(i / float(num_points - 1), normalized=True) for i in range(num_points)]
            points_outside_polygon = [point for point in sample_points if not prohibit_region.contains(point)]
            if len(points_outside_polygon) <= 0:
                continue
            random.shuffle(points_outside_polygon)
            for trie_i in range(self.max_tries):
                if trie_i >= len(points_outside_polygon):
                    break
                point = random.choice(points_outside_polygon)
                point_s = lane_line.project(point)
                point_location, point_heading = self._ma.get_coordinate_and_heading(lane_id, point_s)
                point_shape = agent_type
                front_l = point_shape.length / 2.0
                back_l = point_shape.length / 2.0
                point_polygon, _ = generate_polygon(
                    point_location.x,
                    point_location.y,
                    point_heading,
                    front_l=front_l,
                    back_l=back_l,
                    width=point_shape.width
                )
                if point_polygon.intersects(prohibit_region):
                    continue
                else:
                    # create waypoint
                    waypoint = MotionWaypoint(
                        origin=PositionUnit(
                            lane_id=lane_id,
                            s=point_s,
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
                        origin_speed=0.0,
                        perturb_speed=0.0,
                        is_junction=self._ma.is_junction_lane(lane_id)
                    )
                    return waypoint
        return None

    def _sample_follow_lane_waypoint(self,
                                     last_waypoint: MotionWaypoint,
                                     delta_t: float,
                                     prohibit_region: Polygon,
                                     agent_type: AgentTypeClass,
                                     is_neighbor: bool) -> Optional[MotionWaypoint]:

        last_lane_id = last_waypoint.lane_id
        last_lane_s = last_waypoint.s
        # last_speed = last_waypoint.speed
        last_lane_length = self._ma.get_lane_length(last_lane_id)

        # 1. checking follow lanes
        if is_neighbor:
            travel_range = float(np.clip(random.uniform(5.0, 15.0), 0.0, 20.0)) * delta_t
        else:
            travel_range = float(np.clip(random.uniform(1.0, 20.0), 0.0, 20.0)) * delta_t
        next_long_s = last_lane_s + travel_range

        # 2. find segments
        points_segments = list()
        last_point = Point([last_waypoint.x, last_waypoint.y])
        no_succ = False
        segment_start_s = last_lane_s
        segment_start_travel_s = 0.0
        while next_long_s > last_lane_length:
            next_long_s -= last_lane_length
            point, _heading = self._ma.get_coordinate_and_heading(last_lane_id, last_lane_length)
            line_segment = LineString([[last_point.x, last_point.y], [point.x, point.y]])
            points_segments.append((last_lane_id, segment_start_s, segment_start_travel_s, line_segment))
            last_point = point
            segment_start_s = 0.0
            segment_start_travel_s += line_segment.length

            last_lane_successors = self._ma.get_successor_lanes(last_lane_id, driving_only=True)
            if len(last_lane_successors) <= 0:
                no_succ = True
                break
            else:
                last_lane_id = random.choice(last_lane_successors)
                last_lane_length = self._ma.get_lane_length(last_lane_id)

        if not no_succ:
            point, _heading = self._ma.get_coordinate_and_heading(last_lane_id, next_long_s)
            line_segment = LineString([[last_point.x, last_point.y], [point.x, point.y]])
            points_segments.append((last_lane_id, segment_start_s, segment_start_travel_s, line_segment))

        # 3. find points
        stop_flag = False
        points_outside_polygon = list()
        for i, segment in enumerate(points_segments):
            segment_lane_id = segment[0]
            segment_start_s = segment[1]
            segment_start_travel_s = segment[2]
            segment_line = segment[3]
            segment_line_length = segment_line.length
            num_points = int(segment_line_length / 1.0) + 1
            num_points = max(num_points, 2)
            sample_points = [segment_line.interpolate(i / float(num_points - 1), normalized=True) for i in range(num_points)]
            for point in sample_points:
                if not prohibit_region.contains(point):
                    points_outside_polygon.append((segment_lane_id, segment_start_s, segment_start_travel_s, segment_line, point))
                else:
                    stop_flag = True
                    break
            if stop_flag:
                break

        # 4. return None
        if len(points_outside_polygon) <= 0:
            return None

        # 5. check points
        random.shuffle(points_outside_polygon)
        for i in range(self.max_tries):
            if i >= len(points_outside_polygon):
                break
            sample_point_tuple = points_outside_polygon[i]
            segment_lane_id = sample_point_tuple[0]
            segment_start_s = sample_point_tuple[1]
            segment_start_travel_s = sample_point_tuple[2]
            segment_line: LineString = sample_point_tuple[3]
            point = sample_point_tuple[4]
            # split_point = segment_line.interpolate(segment_line.project(point))
            segment_s = segment_line.project(point)

            point_s = segment_s + segment_start_s
            point_travel_s = segment_start_travel_s + segment_s
            point_location, point_heading = self._ma.get_coordinate_and_heading(segment_lane_id, point_s)
            point_shape = agent_type
            front_l = point_shape.length / 2.0
            back_l = point_shape.length / 2.0
            point_polygon, _ = generate_polygon(
                point_location.x,
                point_location.y,
                point_heading,
                front_l=front_l,
                back_l=back_l,
                width=point_shape.width
            )
            if point_polygon.intersects(prohibit_region):
                continue
            else:
                speed = point_travel_s / float(delta_t)
                # create waypoint
                waypoint = MotionWaypoint(
                    origin=PositionUnit(
                        lane_id=segment_lane_id,
                        s=point_s,
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
                    origin_speed=speed,
                    perturb_speed=0.0,
                    is_junction=self._ma.is_junction_lane(segment_lane_id)
                )
                return waypoint
        return None

    def _sample_neighbor_waypoint(self, last_waypoint: MotionWaypoint) -> Optional[MotionWaypoint]:

        last_lane_id = last_waypoint.lane_id
        last_lane_neighbors = self._ma.get_neighbors(last_lane_id, direct = 'forward', side='both', driving_only=False)
        if len(last_lane_neighbors) <= 0:
            return None

        last_waypoint_neighbor = copy.deepcopy(last_waypoint)
        last_point = Point([last_waypoint.x, last_waypoint.y])
        last_neighbor_id = random.choice(last_lane_neighbors)
        last_neighbor_line = self._ma.get_lane_central_curve(last_neighbor_id)
        last_neighbor_s = last_neighbor_line.project(last_point)

        last_neighbor_point, last_neighbor_heading = self._ma.get_coordinate_and_heading(last_neighbor_id, last_neighbor_s)
        last_waypoint_neighbor.origin.lane_id = last_neighbor_id
        last_waypoint_neighbor.origin.s = last_neighbor_s
        last_waypoint_neighbor.origin.x = last_neighbor_point.x
        last_waypoint_neighbor.origin.y = last_neighbor_point.y
        last_waypoint_neighbor.origin.heading = last_neighbor_heading

        return last_waypoint_neighbor

    def search(self,
               last_waypoint: MotionWaypoint,
               delta_t: float,
               prohibit_region: Polygon,
               agent_type: AgentTypeClass) -> Optional[MotionWaypoint]:

        # 1. if start waypoint
        if last_waypoint is None:
            return self._sample_start_waypoint(prohibit_region, agent_type)

        # 2. confirm the same lane or next successor lane
        next_waypoint = self._sample_follow_lane_waypoint(last_waypoint, delta_t, prohibit_region, agent_type, is_neighbor=False)
        if next_waypoint is not None:
            return next_waypoint

        # 3. follow is none, try change lane
        last_neighbor_waypoint = self._sample_neighbor_waypoint(last_waypoint)
        if last_neighbor_waypoint is not None:
            next_waypoint = self._sample_follow_lane_waypoint(last_neighbor_waypoint, delta_t, prohibit_region, agent_type, is_neighbor=True)
            return next_waypoint # None or waypoint

        return None


class WalkerWaypointSearcher:
    """
    Search NPC vehicle route according to:
    1. lanes vehicle
    2. prohibited region
    """

    def __init__(self, region: RESection, max_tries: int):

        self.crosswalks = region.crosswalks

        self._ma = MapParser.get_instance()
        self.max_tries = max_tries

    def _sample_start_waypoint(self,
                               prohibit_region: Polygon,
                               agent_type: AgentTypeClass) -> Optional[WalkerMotionWaypoint]:

        crosswalks = copy.deepcopy(self.crosswalks)
        random.shuffle(crosswalks)
        for i, cw_id in enumerate(crosswalks):
            cw_boundary = self._ma.get_crosswalk_polygon_by_id(cw_id)
            coords_list = list(cw_boundary.exterior.coords)
            lane_line = LineString(coords_list[:-1])
            lane_line_length = lane_line.length
            num_points = int(lane_line_length / 1.0) + 1
            num_points = max(num_points, 2)
            sample_points = [lane_line.interpolate(i / float(num_points - 1), normalized=True) for i in range(num_points)]
            points_outside_polygon = [point for point in sample_points if not prohibit_region.contains(point)]
            if len(points_outside_polygon) <= 0:
                continue
            random.shuffle(points_outside_polygon)
            for trie_i in range(self.max_tries):
                if trie_i >= len(points_outside_polygon):
                    break
                point = random.choice(points_outside_polygon)
                point_s = lane_line.project(point)
                point_location, point_heading = self._ma.get_coordinate_and_heading_any(lane_line, point_s)
                point_shape = agent_type
                front_l = point_shape.length / 2.0
                back_l = point_shape.length / 2.0
                point_polygon, _ = generate_polygon(
                    point_location.x,
                    point_location.y,
                    point_heading,
                    front_l=front_l,
                    back_l=back_l,
                    width=point_shape.width
                )
                if point_polygon.intersects(prohibit_region):
                    continue
                else:
                    # create waypoint
                    waypoint = WalkerMotionWaypoint(
                        origin=PositionUnit(
                            lane_id=cw_id,
                            s=point_s,
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
                        origin_speed=0.0,
                        perturb_speed=0.0,
                        crosswalk_id=cw_id
                    )
                    return waypoint
        return None

    def _sample_follow_lane_waypoint(self,
                                     last_waypoint: WalkerMotionWaypoint,
                                     delta_t: float,
                                     prohibit_region: Polygon,
                                     agent_type: AgentTypeClass) -> Optional[WalkerMotionWaypoint]:

        last_cw_id = last_waypoint.crosswalk_id
        cw_boundary = self._ma.get_crosswalk_polygon_by_id(last_cw_id)
        coords_list = list(cw_boundary.exterior.coords)
        lane_line = LineString(coords_list[:-1])
        last_lane_length = lane_line.length

        last_lane_s = last_waypoint.s

        # 1. checking follow lanes
        travel_range = random.uniform(1.0, 3.0) * delta_t
        next_long_s = last_lane_s + travel_range

        # 2. find segments
        points_segments = list()
        last_point = Point([last_waypoint.x, last_waypoint.y])
        segment_start_s = last_lane_s
        segment_start_travel_s = 0.0
        while next_long_s > last_lane_length:
            next_long_s -= last_lane_length
            point, _heading = self._ma.get_coordinate_and_heading_any(lane_line, last_lane_length)
            line_segment = LineString([[last_point.x, last_point.y], [point.x, point.y]])
            points_segments.append((last_cw_id, segment_start_s, segment_start_travel_s, line_segment))
            last_point = point
            segment_start_s = 0.0
            segment_start_travel_s += line_segment.length

        point, _heading = self._ma.get_coordinate_and_heading_any(lane_line, next_long_s)
        line_segment = LineString([[last_point.x, last_point.y], [point.x, point.y]])
        points_segments.append((last_cw_id, segment_start_s, segment_start_travel_s, line_segment))

        # 3. find points
        stop_flag = False
        points_outside_polygon = list()
        for i, segment in enumerate(points_segments):
            segment_lane_id = segment[0]
            segment_start_s = segment[1]
            segment_start_travel_s = segment[2]
            segment_line = segment[3]
            segment_line_length = segment_line.length
            num_points = int(segment_line_length / 1.0) + 1
            num_points = max(num_points, 2)
            sample_points = [segment_line.interpolate(i / float(num_points - 1), normalized=True) for i in range(num_points)]
            for point in sample_points:
                if not prohibit_region.contains(point):
                    points_outside_polygon.append((segment_lane_id, segment_start_s, segment_start_travel_s, segment_line, point))
                else:
                    stop_flag = True
                    break
            if stop_flag:
                break

        # 4. return None
        if len(points_outside_polygon) <= 0:
            return None

        # 5. check points
        random.shuffle(points_outside_polygon)
        for i in range(self.max_tries):
            if i >= len(points_outside_polygon):
                break
            sample_point_tuple = points_outside_polygon[i]
            segment_lane_id = sample_point_tuple[0]
            segment_start_s = sample_point_tuple[1]
            segment_start_travel_s = sample_point_tuple[2]
            segment_line: LineString = sample_point_tuple[3]
            point = sample_point_tuple[4]
            # split_point = segment_line.interpolate(segment_line.project(point))
            segment_s = segment_line.project(point)

            point_s = segment_s + segment_start_s
            point_travel_s = segment_start_travel_s + segment_s
            point_location, point_heading = self._ma.get_coordinate_and_heading_any(lane_line, point_s)
            point_shape = agent_type
            front_l = point_shape.length / 2.0
            back_l = point_shape.length / 2.0
            point_polygon, _ = generate_polygon(
                point_location.x,
                point_location.y,
                point_heading,
                front_l=front_l,
                back_l=back_l,
                width=point_shape.width
            )
            if point_polygon.intersects(prohibit_region):
                continue
            else:
                speed = point_travel_s / float(delta_t)
                # create waypoint
                waypoint = WalkerMotionWaypoint(
                    origin=PositionUnit(
                        lane_id=segment_lane_id,
                        s=point_s,
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
                    origin_speed=speed,
                    perturb_speed=0.0,
                    crosswalk_id=segment_lane_id
                )
                return waypoint
        return None

    def search(self,
               last_waypoint: WalkerMotionWaypoint,
               delta_t: float,
               prohibit_region: Polygon,
               agent_type: AgentTypeClass) -> Optional[WalkerMotionWaypoint]:

        # 1. if start waypoint
        if last_waypoint is None:
            return self._sample_start_waypoint(prohibit_region, agent_type)

        # 2. confirm the same lane or next successor lane
        next_waypoint = self._sample_follow_lane_waypoint(last_waypoint, delta_t, prohibit_region, agent_type)
        return next_waypoint
