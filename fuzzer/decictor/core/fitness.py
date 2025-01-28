import math
from typing import TypeVar

import numpy as np
from shapely.geometry import Point, LineString
from scipy.spatial.distance import directed_hausdorff
from fuzzer.common.seed_wrapper import SeedWrapper
from apollo.map_parser import MapParser

SeedWrapperClass = TypeVar('SeedWrapperClass', bound=SeedWrapper)

class OptimalFitness:

    def __init__(self,
                 refer_seed: SeedWrapperClass,
                 mode: str,
                 line_unit: float):
        self.refer_seed = refer_seed
        self.mode = mode # [behavior, path, both]
        self.line_unit = line_unit

        # compute lane_polygons
        self._ma = MapParser.get_instance()
        self.lane_polygons = dict()
        for lane_id in refer_seed.scenario.region.lanes_ego:
            self.lane_polygons[lane_id] = self._ma.get_lane_polygon(lane_id)

        self.refer_behaviors = dict()
        refer_attributes = refer_seed.record.ego.attributes
        for lane_id, lane_polygon in self.lane_polygons.items():
            for attr in refer_attributes:
                attr_point = Point([attr[0], attr[1]])
                if lane_polygon.contains(attr_point):
                    if lane_id not in self.refer_behaviors.keys():
                        self.refer_behaviors[lane_id] = list()
                    self.refer_behaviors[lane_id].append(attr[2:]) # heading, speed, acceleration

    def _calculate_behavior(self, seed: SeedWrapperClass) -> float:
        # TODO: calculate behavior distance on spatial space
        # speed_range = [0, 12]  # m/s
        # acc_range = [0, 10]
        # heading_range = [-math.pi, math.pi]
        # High effect on heading rather than speeding and acceleration

        def linear_kernel(x, y):
            """
            Compute the linear kernel between two sets of vectors.
            """
            return np.dot(x, y.T)

        def compute_mmd(X, Y):
            """
            Compute the Maximum Mean Discrepancy (MMD) between two samples, X and Y,
            using a linear kernel.
            """
            XX = linear_kernel(X, X)
            YY = linear_kernel(Y, Y)
            XY = linear_kernel(X, Y)

            return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

        range_min = np.array([-math.pi, 0, 0])
        range_max = np.array([math.pi, 12, 10])
        weight = np.array([1.0, 0.5, 0.5])

        fitness_lst = list()
        seed_attributes = seed.record.ego.attributes
        for lane_id, refer_lane_behavior in self.refer_behaviors.items():
            lane_polygon = self.lane_polygons[lane_id]
            seed_lane_attribute = list()
            for attr in seed_attributes:
                attr_point = Point([attr[0], attr[1]])
                if lane_polygon.contains(attr_point):
                    seed_lane_attribute.append(attr[2:])
            if len(seed_lane_attribute) <= 0:
                norm_seed_behavior = np.array([[0.0, 0.0, 0.0]])
            else:
                seed_lane_attribute = np.array(seed_lane_attribute)
                norm_seed_behavior = (seed_lane_attribute - range_min) / (range_max - range_min)

            refer_lane_attribute = np.array(refer_lane_behavior)
            norm_seed_behavior = norm_seed_behavior * weight
            norm_refer_behavior = ((refer_lane_attribute - range_min) / (range_max - range_min)) * weight
            mmd_value = compute_mmd(norm_seed_behavior, norm_refer_behavior)
            fitness_lst.append(mmd_value * 10)
        distance = np.average(fitness_lst)
        return float(distance)

    def _calculate_path(self, seed: SeedWrapperClass) -> float:

        def resample_linestring_fixed_distance(line: np.ndarray, distance: float) -> np.ndarray:
            line = LineString(line)
            num_segments = int(line.length // distance)
            resampled_points = [line.interpolate(distance * i) for i in range(num_segments + 1)]

            # Add the last point if it's not included
            if line.length % distance != 0:
                resampled_points.append(line.interpolate(1, normalized=True))
            return np.array([(point.x, point.y) for point in resampled_points])

        ego_trace = np.array(seed.record.ego.trace_pts)
        refer_trace = np.array(self.refer_seed.record.ego.trace_pts)

        ego_start_point = ego_trace[0]
        refer_start_point = refer_trace[0]

        ego_trace = ego_trace - ego_start_point
        refer_trace = refer_trace - refer_start_point

        ego_trace = resample_linestring_fixed_distance(ego_trace, self.line_unit)
        refer_trace = resample_linestring_fixed_distance(refer_trace, self.line_unit)
        frechet_dist = directed_hausdorff(ego_trace, refer_trace)[0]
        return float(frechet_dist)

    def calculate(self, seed: SeedWrapperClass) -> float:
        seed.update_record()
        if self.mode == 'behavior':
            return self._calculate_behavior(seed)
        elif self.mode == 'path':
            return self._calculate_path(seed)
        else:
            return self._calculate_behavior(seed) + self._calculate_path(seed)