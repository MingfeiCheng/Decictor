import os
import copy
import math
import pickle
import random
import networkx as nx

from loguru import logger
from collections import defaultdict
from typing import List, Set, Tuple, Any
from shapely.geometry import LineString, Point, Polygon

from apollo_modules.modules.map.proto.map_crosswalk_pb2 import Crosswalk
from apollo_modules.modules.map.proto.map_junction_pb2 import Junction
from apollo_modules.modules.map.proto.map_lane_pb2 import Lane
from apollo_modules.modules.map.proto.map_pb2 import Map
from apollo_modules.modules.map.proto.map_signal_pb2 import Signal
from apollo_modules.modules.common.proto.geometry_pb2 import PointENU

def load_hd_map(filename: str):
    apollo_map = Map()
    f = open(filename, 'rb')
    apollo_map.ParseFromString(f.read())
    f.close()
    return apollo_map

class MapParser:
    """
    Class to load and parse HD Map from project folder

    :param str filename: filename of the HD Map
    """

    __junctions: dict
    __signals: dict
    __lanes: dict
    __crosswalk: dict
    __overlaps: dict

    __lanes_at_junction: dict

    __instance = None

    def __init__(self):
        """
        Constructor
        """
        self.__junctions = None
        self.__signals = None
        self.__lanes = None
        self.__crosswalk = None

        self.__lanes_at_junction = None

        self.__instance = None

    @staticmethod
    def get_instance() -> 'MapParser':
        """
        Get the singleton instance of MapParser
        """
        if MapParser.__instance is None:
            logger.info('Map instance is None, call init_instance_from_path first!')
            raise RuntimeError
        return MapParser.__instance

    ##### Added #####
    def parse_from_source(self, source_file: str):
        logger.info(f"Load map from source: {source_file}")

        __map = load_hd_map(source_file)

        # load junctions
        self.__junctions = dict()
        for junc in __map.junction:
            self.__junctions[junc.id.id] = junc

        # load signals
        self.__signals = dict()
        for sig in __map.signal:
            self.__signals[sig.id.id] = sig

        # load lanes
        self.__lanes = dict()
        for l in __map.lane:
            self.__lanes[l.id.id] = l

        # load crosswalks
        self.__crosswalk = dict()
        for cw in __map.crosswalk:
            self.__crosswalk[cw.id.id] = cw

        self.parse_relations()

        MapParser.__instance = self

    def export(self, file_folder: str):
        save_data = {
            'junctions': self.__junctions,
            'signals': self.__signals,
            'lanes': self.__lanes,
            'crosswalk': self.__crosswalk,
            'lanes_at_junction': self.__lanes_at_junction,
        }

        for k, y in save_data['junctions'].items():
            save_data['junctions'][k] = y.SerializeToString()

        for k, y in save_data['signals'].items():
            save_data['signals'][k] = y.SerializeToString()

        for k, y in save_data['lanes'].items():
            save_data['lanes'][k] = y.SerializeToString()

        for k, y in save_data['crosswalk'].items():
            save_data['crosswalk'][k] = y.SerializeToString()

        save_file = os.path.join(file_folder, 'map.pickle')
        with open(save_file, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Map saved to {save_file}")

    def load_from_file(self, file_folder: str):

        map_file = os.path.join(file_folder, 'map.pickle')
        with open(map_file, 'rb') as f:
            attributes_dict = pickle.load(f)

        """
        save_data = {
            'junctions': self.__junctions,
            'signals': self.__signals,
            'lanes': self.__lanes,
            'crosswalk': self.__crosswalk,
            'lanes_at_junction': self.__lanes_at_junction,
        }
        """
        self.__junctions = attributes_dict['junctions']
        self.__signals = attributes_dict['signals']
        self.__lanes = attributes_dict['lanes']
        self.__crosswalk = attributes_dict['crosswalk']
        self.__lanes_at_junction = attributes_dict['lanes_at_junction']

        for k, y in self.__junctions.items():
            junc = Junction()
            junc.ParseFromString(y)
            self.__junctions[k] = junc

        for k, y in self.__signals.items():
            sig = Signal()
            sig.ParseFromString(y)
            self.__signals[k] = sig

        for k, y in self.__lanes.items():
            lan = Lane()
            lan.ParseFromString(y)
            self.__lanes[k] = lan

        for k, y in self.__crosswalk.items():
            cw = Crosswalk()
            cw.ParseFromString(y)
            self.__crosswalk[k] = cw

        self.parse_relations()

        MapParser.__instance = self
        logger.info(f"Load map from {map_file}")

    def parse_relations(self):
        """
        Parse relations between signals and junctions,
        lanes and junctions, and lanes and signals
        """
        # load lanes at junction
        self.__lanes_at_junction = defaultdict(list)
        for lank, lanv in self.__lanes.items():
            for junk, junv in self.__junctions.items():
                if self.__is_overlap(lanv, junv):
                    self.__lanes_at_junction[junk].append(lank)


    def __is_overlap(self, obj1, obj2):
        """
        Check if 2 objects (e.g., lanes, junctions) have overlap

        :param any obj1: left hand side
        :param any obj2: right hand side
        """
        oid1 = set([x.id for x in obj1.overlap_id])
        oid2 = set([x.id for x in obj2.overlap_id])
        return oid1 & oid2 != set()

    def is_conflict_lanes(self, lane_id1: List[str], lane_id2: List[str]) -> bool:
        """
        Check if 2 groups of lanes intersect with each other

        :param List[str] lane_id1: list of lane ids
        :param List[str] lane_id2: another list of lane ids

        :returns: True if at least 1 lane from lhs intersects with another from rhs,
            False otherwise.
        :rtype: bool
        """
        for lid1 in lane_id1:
            for lid2 in lane_id2:
                if lid1 == lid2:
                    continue
                lane1 = self.get_lane_central_curve(lid1)
                lane2 = self.get_lane_central_curve(lid2)
                if lane1.intersects(lane2):
                    return True
        return False

    def get_lane_central_curve(self, lane_id: str) -> LineString:
        """
        Gets the central curve of the lane.

        :param str lane_id: ID of the lane interested in

        :returns: an object representing the lane's central curve
        :rypte: LineString
        """
        lane = self.__lanes[lane_id]
        points = lane.central_curve.segment[0].line_segment
        line = LineString([[x.x, x.y] for x in points.point])
        return line

    def get_lane_boundary_curve(self, lane_id: str) -> Tuple[LineString, LineString]:
        lane = self.__lanes[lane_id]
        points = lane.left_boundary.curve.segment[0].line_segment
        left_line = LineString([[x.x, x.y] for x in points.point])
        points = lane.right_boundary.curve.segment[0].line_segment
        right_line = LineString([[x.x, x.y] for x in points.point])
        return left_line, right_line

    def get_lane_polygon(self, lane_id: str) -> Polygon:
        lane = self.__lanes[lane_id]
        points = lane.left_boundary.curve.segment[0].line_segment
        left_line = [[x.x, x.y] for x in points.point]
        points = lane.right_boundary.curve.segment[0].line_segment
        right_line = [[x.x, x.y] for x in points.point]

        right_line = right_line[::-1]
        lane_boundary = left_line + right_line
        return Polygon(lane_boundary)

    def get_lane_length(self, lane_id: str) -> float:
        """
        Gets the length of the lane.

        :param str lane_id: ID of the lane interested in

        :returns: length of the lane
        :rtype: float
        """
        return self.get_lane_central_curve(lane_id).length

    def get_coordinate_and_heading(self, lane_id: str, s: float) -> Tuple[PointENU, float]:
        """
        Given a lane_id and a point on the lane, get the actual coordinate and the heading
        at that point.

        :param str lane_id: ID of the lane intereted in
        :param float s: meters away from the start of the lane

        :returns: coordinate and heading in a tuple
        :rtype: Tuple[PointENU, float]
        """
        lst = self.get_lane_central_curve(lane_id)  # line string
        ip = lst.interpolate(s)  # a point

        segments = list(map(LineString, zip(lst.coords[:-1], lst.coords[1:])))
        segments.sort(key=lambda x: ip.distance(x))
        line = segments[0]
        x1, x2 = line.xy[0]
        y1, y2 = line.xy[1]

        return PointENU(x=ip.x, y=ip.y), math.atan2(y2 - y1, x2 - x1)

    def get_coordinate_and_heading_any(self, line: LineString, s: float) -> Tuple[PointENU, float]:
        lst = line
        ip = lst.interpolate(s)  # a point
        segments = list(map(LineString, zip(lst.coords[:-1], lst.coords[1:])))
        segments.sort(key=lambda x: ip.distance(x))
        line = segments[0]
        x1, x2 = line.xy[0]
        y1, y2 = line.xy[1]
        return PointENU(x=ip.x, y=ip.y), math.atan2(y2 - y1, x2 - x1)

    def get_junctions(self) -> List[str]:
        """
        Get a list of all junction IDs on the HD Map

        :returns: list of junction IDs
        :rtype: List[str]
        """
        return list(self.__junctions.keys())

    def get_junction_by_id(self, j_id: str) -> Junction:
        """
        Get a specific junction object based on ID

        :param str j_id: ID of the junction interested in

        :returns: junction object
        :rtype: Junction
        """
        return self.__junctions[j_id]

    def get_lanes(self) -> List[str]:
        """
        Get a list of all lane IDs on the HD Map

        :returns: list of lane IDs
        :rtype: List[str]
        """
        return list(self.__lanes.keys())

    def get_lane_by_id(self, l_id: str) -> Lane:
        """
        Get a specific junction object based on ID

        :param str l_id: ID of the lane interested in

        :returns: lane object
        :rtype: Lane
        """
        return self.__lanes[l_id]

    def get_crosswalks(self) -> List[str]:
        """
        Get a list of all crosswalk IDs on the HD Map

        :returns: list of crosswalk IDs
        :rtype: List[str]
        """
        return list(self.__crosswalk.keys())

    def get_crosswalk_by_id(self, cw_id: str) -> Crosswalk:
        """
        Get a specific crosswalk object based on ID

        :param str cw_id: ID of the crosswalk interested in

        :returns: crosswalk object
        :rtype: Crosswalk
        """
        return self.__crosswalk[cw_id]

    def get_crosswalk_polygon_by_id(self, cw_id: str) -> Polygon:
        """
        """
        cw = self.__crosswalk[cw_id]
        cw_polygon_points = cw.polygon.point
        polygon_coords = []
        for point in cw_polygon_points:
            polygon_coords.append([point.x, point.y])
        start_point = polygon_coords[0]
        polygon_coords.append(start_point)
        cw_polygon = Polygon(polygon_coords)
        return cw_polygon

    def get_signals(self) -> List[str]:
        """
        Get a list of all signal IDs on the HD Map

        :returns: list of signal IDs
        :rtype: List[str]
        """
        return list(self.__signals.keys())

    def get_signal_by_id(self, s_id: str) -> Signal:
        """
        Get a specific signal object based on ID

        :param str s_id: ID of the signal interested in

        :returns: signal object
        :rtype: Signal
        """
        return self.__signals[s_id]

    def get_lanes_not_in_junction(self) -> Set[str]:
        """
        Get the set of all lanes that are not in the junction.

        :returns: ID of lanes who is not in a junction
        :rtype: Set[str]
        """
        lanes = set(self.get_lanes())
        for junc in self.__lanes_at_junction:
            jlanes = set(self.__lanes_at_junction[junc])
            lanes = lanes - jlanes
        return lanes


    def get_predecessor_lanes(self, lane_id, driving_only=False) -> List[str]:
        current_lane = self.__lanes[lane_id]
        predecessor_lane_ids = current_lane.predecessor_id
        predecessor_lane_lst = []
        for item in predecessor_lane_ids:
            predecessor_lane_id = item.id
            if driving_only:
                if self.is_driving_lane(predecessor_lane_id):
                    predecessor_lane_lst.append(predecessor_lane_id)
            else:
                predecessor_lane_lst.append(predecessor_lane_id)
        return predecessor_lane_lst

    def get_successor_lanes(self, lane_id, driving_only=False) -> List[str]:
        current_lane = self.__lanes[lane_id]
        successor_lane_ids = current_lane.successor_id
        successor_lane_lst = []
        for item in successor_lane_ids:
            successor_lane_id = item.id
            if driving_only:
                if self.is_driving_lane(successor_lane_id):
                    successor_lane_lst.append(successor_lane_id)
            else:
                successor_lane_lst.append(successor_lane_id)
        return successor_lane_lst

    def get_neighbors(self, lane_id, direct = 'forward', side='left', driving_only=False) -> List:
        assert side in ['left', 'right', 'both']
        assert direct in ['forward', 'reverse', 'both']

        lane = self.__lanes[lane_id]

        # Find forward neighbors
        forward_neighbors = list()
        if (side == 'left' or side == 'both') and (direct == 'forward' or direct == 'both'):
            left_forwards = lane.left_neighbor_forward_lane_id
            for lf in left_forwards:
                lf_id = lf.id
                if not driving_only:
                    forward_neighbors.append(lf_id)
                else:
                    if self.is_driving_lane(lf_id):
                        forward_neighbors.append(lf_id)

        if (side == 'right' or side == 'both') and (direct == 'forward' or direct == 'both'):
            right_forwards = lane.right_neighbor_forward_lane_id
            for rf in right_forwards:
                rf_id = rf.id
                if not driving_only:
                    forward_neighbors.append(rf_id)
                else:
                   if self.is_driving_lane(rf_id):
                        forward_neighbors.append(rf_id)

        # Find reverse neighbors
        reverse_neighbors = list()
        if (side == 'left' or side == 'both') and (direct == 'reverse' or direct == 'both'):
            left_reverses = lane.left_neighbor_reverse_lane_id
            for lr in left_reverses:
                lr_id = lr.id
                if not driving_only:
                    reverse_neighbors.append(lr_id)
                else:
                    if self.is_driving_lane(lr_id):
                        reverse_neighbors.append(lr_id)

        if (side == 'right' or side == 'both') and (direct == 'reverse' or direct == 'both'):
            right_reverses = lane.right_neighbor_reverse_lane_id
            for rr in right_reverses:
                rr_id = rr.id
                if not driving_only:
                    reverse_neighbors.append(rr_id)
                else:
                    if self.is_driving_lane(rr_id):
                        reverse_neighbors.append(rr_id)

        neighbors = forward_neighbors + reverse_neighbors
        return neighbors

    def get_wide_neighbors(self, lane_id, direct = 'forward', side='left', driving_only=False) -> List:
        current_lanes = [lane_id]
        last_lanes = list()
        while len(current_lanes) != len(last_lanes):
            last_lanes = copy.deepcopy(current_lanes)
            for _id in last_lanes:
                current_lanes += self.get_neighbors(_id, direct, side, driving_only)
            current_lanes = list(set(current_lanes))

        return current_lanes

    def get_coordinate(self, lane_id: str, s: float, l: float):
        """
        Given a lane_id and a point on the lane, get the actual coordinate and the heading
        at that point.
        """

        def right_rotation(coord, theta):
            """
            theta : degree
            """
            # theta = math.radians(theta)
            x_o = coord[1]
            y_o = coord[0]
            x_r = x_o * math.cos(theta) - y_o * math.sin(theta)
            y_r = x_o * math.sin(theta) + y_o * math.cos(theta)
            return [y_r, x_r]

        lst = self.get_lane_central_curve(lane_id)  # line string
        # logger.debug('s: {}', s)
        ip = lst.interpolate(s)  # a point

        segments = list(map(LineString, zip(lst.coords[:-1], lst.coords[1:])))
        # logger.debug('ip: type {} {}', type(ip), ip)
        segments.sort(key=lambda t: ip.distance(t))
        line = segments[0]
        x1, x2 = line.xy[0]
        y1, y2 = line.xy[1]

        heading = math.atan2(y2 - y1, x2 - x1)

        init_vector = [1, 0]
        right_vector = right_rotation(init_vector, -(heading - math.radians(90.0)))
        x = ip.x + right_vector[0] * l
        y = ip.y + right_vector[1] * l
        return x, y, heading

    def get_junction_by_lane_id(self, lane_id):
        """
        Return the junction id where the lane in the junction
        """
        for k, v in self.__lanes_at_junction.items():
            if lane_id in v:
                return k
        return None

    def get_lane_width(self, lane_id):
        return self.__lanes[lane_id].width

    def get_lanes_in_junction_id(self, junc_id):
        return list(set(self.__lanes_at_junction[junc_id]))

    def is_junction_lane(self, lane_id) -> bool:
        """
        Return the junction id where the lane in the junction
        """
        for k, v in self.__lanes_at_junction.items():
            if lane_id in v:
                return True
        return False

    def is_driving_lane(self, lane_id) -> bool:
        lane_obj = self.get_lane_by_id(lane_id)
        lane_type = lane_obj.type
        if lane_type == lane_obj.LaneType.CITY_DRIVING:
            return True
        else:
            return False


    def get_lanes_from_trace(self,
                             start_lane,
                             end_lane,
                             ego_trace: LineString) -> Tuple[List, List]:

        def find_path(graph, start, end, t_path):
            t_path = t_path + [start]
            # logger.debug('t_path: {}', t_path)
            if start == end:
                return t_path
            if start not in graph:
                return None
            # logger.debug('graph[{}]: {}', start, graph[start])
            for node in graph[start]:
                if node not in t_path:
                    # logger.debug('node: {}', node)
                    newpath = find_path(graph, node, end, t_path)
                    if newpath:
                        return newpath
            return None

        ### Step1: extract line regions
        lane_pool = [start_lane, end_lane]
        while True:
            last_lane_pool = copy.deepcopy(lane_pool)
            next_pool = list()
            for lane_id in lane_pool:
                next_pool += self.get_wide_neighbors(lane_id, direct='forward', side='both', driving_only=True)
                next_pool += self.get_successor_lanes(lane_id, driving_only=True)
                next_pool += self.get_predecessor_lanes(lane_id, driving_only=True)
            next_pool = list(set(next_pool))

            next_pool_filter = list()
            for lane_id in next_pool:
                lane_polygon = self.get_lane_polygon(lane_id)
                if ego_trace.intersects(lane_polygon):
                    next_pool_filter.append(lane_id)
            next_pool_filter = list(set(next_pool_filter))

            lane_pool += next_pool_filter
            lane_pool = list(set(lane_pool))

            if len(lane_pool) == len(last_lane_pool):
                break

        ### Step2: construct connection in lane pool
        lane_graph = dict()
        lane_pool = list(set(lane_pool))
        for lane_id in lane_pool:
            direct_connections = list()
            direct_connections += self.get_neighbors(lane_id, direct='forward', side='both', driving_only=True)
            direct_connections += self.get_successor_lanes(lane_id, driving_only=True)
            lane_graph[lane_id] = list(set(direct_connections))

        trace_path = list()
        trace_path = find_path(lane_graph, start_lane, end_lane, trace_path)
        return trace_path, lane_pool


    def get_random_route_by_cover_lane(self,
                                       cover_lane: Any,
                                       prev_depth: int = 1,
                                       next_depth: int = 3) -> List:

        trace = [cover_lane]
        for i in range(prev_depth):
            last_lane = trace[0]
            last_lane_predecessors = self.get_predecessor_lanes(last_lane, driving_only=False)
            if len(last_lane_predecessors) == 0:
                break
            pred_lane = random.choice(last_lane_predecessors)
            trace.insert(0, pred_lane)

        for i in range(next_depth):
            last_lane = trace[-1]
            last_lane_successors = self.get_successor_lanes(last_lane, driving_only=False)
            if len(last_lane_successors) == 0:
                break
            succ_lane = random.choice(last_lane_successors)
            trace.append(succ_lane)
        return trace

    def get_random_route_by_prev_lanes(self,
                                       lane_pool: List,
                                       prev_depth: int = 1,
                                       next_depth: int = 3) -> List:
        tmp_lane = random.choice(lane_pool)
        trace = [tmp_lane]
        for i in range(prev_depth):
            last_lane = trace[0]
            last_lane_predecessors = self.get_predecessor_lanes(last_lane, driving_only=False)
            if len(last_lane_predecessors) == 0:
                break
            lane_candidate = list()
            for lane_id in last_lane_predecessors:
                if lane_id in lane_pool:
                    lane_candidate.append(lane_id)
            lane_candidate = list(set(lane_candidate))
            if len(lane_candidate) == 0:
                break
            pred_lane = random.choice(lane_candidate)
            trace.insert(0, pred_lane)

        count = 0
        junction_id = None
        while count < next_depth:
            last_lane = trace[-1]
            last_lane_successors = self.get_successor_lanes(last_lane, driving_only=False)
            if len(last_lane_successors) == 0:
                break
            succ_lane = random.choice(last_lane_successors)
            trace.append(succ_lane)
            if self.is_junction_lane(succ_lane):
                if junction_id is None:
                    junction_id = self.get_junction_by_lane_id(succ_lane)
                if self.get_junction_by_lane_id(succ_lane) == junction_id:
                    continue
                else:
                    count += 1
            else:
                count += 1
        return trace

    def get_random_route_from_start_lane(self,
                                         start_lane: Any,
                                         next_depth: int = 3) -> List:

        trace = [start_lane]
        count = 0
        while count < next_depth:
            last_lane = trace[-1]
            last_lane_successors = self.get_successor_lanes(last_lane, driving_only=False)
            if len(last_lane_successors) == 0:
                break
            succ_lane = random.choice(last_lane_successors)
            trace.append(succ_lane)
            count += 1
        return trace

    def get_random_changing_route_from_start_lane(self,
                                                  start_lane: Any,
                                                  next_depth: int = 3,
                                                  lane_change_limit: int = 1) -> List:

        trace = [start_lane]
        count = 0
        lane_change_count = 0
        while count < next_depth:
            last_lane = trace[-1]
            if self.is_junction_lane(last_lane):
                last_lane_successors = self.get_successor_lanes(last_lane, driving_only=True)
                if len(last_lane_successors) == 0:
                    break
                succ_lane = random.choice(last_lane_successors)
                trace.append(succ_lane)
                count += 1
            else:
                last_lane_neighbors = self.get_neighbors(last_lane, direct='forward', side='both', driving_only=True)
                if len(last_lane_neighbors) == 0:
                    last_lane_successors = self.get_successor_lanes(last_lane, driving_only=True)
                    if len(last_lane_successors) == 0:
                        break
                    succ_lane = random.choice(last_lane_successors)
                    trace.append(succ_lane)
                    count += 1
                else:
                    if lane_change_count >= lane_change_limit:
                        last_lane_successors = self.get_successor_lanes(last_lane, driving_only=True)
                        if len(last_lane_successors) == 0:
                            break
                        succ_lane = random.choice(last_lane_successors)
                        trace.append(succ_lane)
                        count += 1
                    else:
                        if random.random() > 0.5:
                            last_lane_successors = self.get_successor_lanes(last_lane, driving_only=True)
                            if len(last_lane_successors) == 0:
                                break
                            succ_lane = random.choice(last_lane_successors)
                            trace.append(succ_lane)
                            count += 1
                        else:
                            last_lane_neighbor = random.choice(last_lane_neighbors)
                            last_lane_successors = self.get_successor_lanes(last_lane_neighbor, driving_only=True)
                            if len(last_lane_successors) == 0:
                                break
                            succ_lane = random.choice(last_lane_successors)
                            trace.append(succ_lane)
                            count += 1
                            lane_change_count += 1
        return trace

    def get_waypoint_s_for_lane(self,
                               lane_id: str,
                               waypoint_interval: float) -> List[float]:
        """
        Generate initial waypoints for a NPC
        reduce waypoint number
        """
        lane_length = self.get_lane_length(lane_id)

        s_lst = [1.0]
        while True:
            last_s = s_lst[-1]
            next_s = last_s + waypoint_interval
            if next_s < lane_length:
                s_lst.append(next_s)
            else:
                s_lst.append(lane_length)
                break

        return s_lst

    def get_next_waypoint(self, radius: float, lane_id: str, s: float) -> List:
        next_s = s + radius
        lane_length = self.get_lane_length(lane_id)
        next_waypoints = list()

        if lane_length < next_s:
            next_lane_pool = self.get_successor_lanes(lane_id, driving_only=True)
            for next_lane_id in next_lane_pool:
                next_waypoints.append((next_lane_id, next_s - lane_length))
        else:
            next_waypoints.append((lane_id, next_s))

        return next_waypoints
