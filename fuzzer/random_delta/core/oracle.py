import numpy as np

from typing import TypeVar, List, Tuple, Any
from loguru import logger
from scipy import interpolate
from shapely.geometry import Polygon, LineString

from apollo.map_parser import MapParser
from fuzzer.common.seed_wrapper import SeedWrapper
from tools.utils import generate_polygon

SeedWrapperClass = TypeVar('SeedWrapperClass', bound=SeedWrapper)

class OptimalOracle(object):

    def __init__(self,
                 refer_seed: SeedWrapperClass,
                 optimal_threshold: float,
                 grid_unit: float):
        self.refer_seed = refer_seed
        self.refer_record = refer_seed.record
        # inner parameters
        self.optimal_threshold = optimal_threshold
        self.grid_unit = grid_unit
        # preprocess for acceleration
        # if self.refer_record.ego.lane_grid is None:
        self.refer_t = None
        self.f_x = None
        self.f_y = None
        self.f_heading = None
        self._preprocess()

    def _preprocess(self):
        # find refer grid
        # 1. extract trace lanes
        refer_ego_lanes = self.refer_seed.scenario.region.lanes_ego

        # 2. split refer lanes with trace
        ma = MapParser.get_instance()
        refer_lane_grid = dict()
        for lane_id in refer_ego_lanes:  # change to only ego lanes
            # for trace difference
            refer_lane_grid[lane_id] = list()
            lane_left_boundary, lane_right_boundary = ma.get_lane_boundary_curve(lane_id)
            lane_line_length = ma.get_lane_length(lane_id)
            num_points = int(lane_line_length / self.grid_unit) + 1
            num_points = max(num_points, 2)
            left_sample_points = [lane_left_boundary.interpolate(i / float(num_points - 1), normalized=True) for i in range(num_points)]
            right_sample_points = [lane_right_boundary.interpolate(i / float(num_points - 1), normalized=True) for i in range(num_points)]

            assert len(left_sample_points) == len(right_sample_points)

            for i in range(len(left_sample_points)):
                refer_lane_grid[lane_id].append(LineString([left_sample_points[i], right_sample_points[i]]))
        self.refer_record.ego.lane_grid = refer_lane_grid

        # 2. calculate grids for refer ego
        refer_ego_trace: LineString = LineString(self.refer_seed.record.ego.trace_pts)
        refer_lane_occupancy = dict()
        for lane_id, line_lst in refer_lane_grid.items():
            lane_grid_num = len(line_lst)
            if lane_grid_num < 2:
                continue
            lane_grid_nonzero_count = 0
            lane_occupancy = np.zeros(lane_grid_num)
            for line_index in range(lane_grid_num):
                seg_line = line_lst[line_index]
                if refer_ego_trace.intersects(seg_line):
                    lane_occupancy[line_index] = 1
                    lane_grid_nonzero_count += 1
            lane_nonzero_rate = lane_grid_nonzero_count / float(lane_grid_num)
            if lane_nonzero_rate < 0.2:
                continue
            refer_lane_occupancy[lane_id] = lane_occupancy
        #
        #     refer_lane_occupancy[lane_id] = np.zeros(len(line_lst))
        #     for line_index in range(len(line_lst)):
        #         seg_line = line_lst[line_index]
        #         if refer_ego_trace.intersects(seg_line):
        #             refer_lane_occupancy[lane_id][line_index] = 1

        self.refer_record.ego.lane_occupancy = refer_lane_occupancy

        # 3. calculate fx
        refer_ego = self.refer_record.ego.attributes

        refer_ts = np.array(self.refer_record.ego.timestamps)
        refer_xs = list()
        refer_ys = list()
        refer_headings = list()

        for frame_index in range(len(refer_ego)):
            frame = refer_ego[frame_index]
            refer_xs.append(frame[0])  # x
            refer_ys.append(frame[1])  # y
            refer_headings.append(frame[2])  # heading

        self.f_x = interpolate.interp1d(refer_ts, np.array(refer_xs), kind='linear')
        self.f_y = interpolate.interp1d(refer_ts, np.array(refer_ys), kind='linear')
        self.f_heading = interpolate.interp1d(refer_ts, np.array(refer_headings), kind='linear')
        self.refer_t = refer_ts[-1]

    def _is_replay_pass(self, seed: SeedWrapperClass) -> bool:
        """
        replay reference ego behaviors in current record
        """
        # 1. replay with curr env and refer ego
        curr_env = seed.record.env
        curr_env_timestamps = curr_env.timestamps
        curr_env_participants = curr_env.participants

        for frame_index, frame_time in enumerate(curr_env_timestamps):
            # ignore self.refer_t > last_current_time
            if frame_time >= self.refer_t:
                break
            ##### extract prev ego polygon in curr env #####
            refer_ego_x = float(self.f_x(frame_time))
            refer_ego_y = float(self.f_y(frame_time))
            refer_ego_heading = float(self.f_heading(frame_time))
            # fix this by parameters
            ego_length = seed.scenario.egos.agents[0].agent_type.length
            ego_width = seed.scenario.egos.agents[0].agent_type.width
            ego_back2center = seed.scenario.egos.agents[0].agent_type.back2center
            front_l = ego_length - ego_back2center
            back_l = -1 * ego_back2center
            refer_ego_polygon, _ = generate_polygon(refer_ego_x,
                                                    refer_ego_y,
                                                    refer_ego_heading,
                                                    front_l=front_l,
                                                    back_l=back_l,
                                                    width=ego_width)
            ##### End this part #####

            obs_dict = curr_env_participants[frame_index]
            for obs_id, obs_info in obs_dict.items():
                obs_polygon = Polygon(obs_info['polygon'])
                if refer_ego_polygon.distance(obs_polygon) <= 0.0:
                    return False
        return True

    def _is_non_optimal(self, seed: SeedWrapperClass) -> Tuple[bool, float, Any]:
        # calculate the occupancy for record
        ego_trace: LineString = LineString(seed.record.ego.trace_pts)
        # trace_length_diff = ego_trace.length - self.refer_record.ego_trace.length

        ego_occupancy = dict()
        grid_diff_iou = dict()
        refer_lane_grid = self.refer_record.ego.lane_grid
        refer_occupancy = self.refer_record.ego.lane_occupancy
        max_lane_diff_iou = 0.0
        max_lane_id = ''
        for lane_id in refer_occupancy.keys():
            line_lst = refer_lane_grid[lane_id]
            ego_occupancy[lane_id] = np.zeros(len(line_lst))
            for line_index in range(len(line_lst)):
                seg_line = line_lst[line_index]
                if ego_trace.intersects(seg_line):
                    ego_occupancy[lane_id][line_index] = 1

            diff = np.linalg.norm(refer_occupancy[lane_id] - ego_occupancy[lane_id], ord=1)
            union = np.where((refer_occupancy[lane_id] + ego_occupancy[lane_id]) > 0, 1, 0)
            grid_diff_iou[lane_id] = float(diff / float(np.sum(union) + 1e-5))
            if grid_diff_iou[lane_id] > max_lane_diff_iou:
                max_lane_diff_iou = grid_diff_iou[lane_id]
                max_lane_id = lane_id

        if max_lane_diff_iou > self.optimal_threshold:
            return True, max_lane_diff_iou, max_lane_id
        else:
            return False, max_lane_diff_iou, max_lane_id

    def check(self, seed: SeedWrapperClass) -> List[bool]:
        seed.update_record()
        # 1. check original can pass
        is_replay_pass = self._is_replay_pass(seed) # true: pass, false: not pass

        # 2. check path optimal
        is_non_optimal, diff_iou_value, lane_id = self._is_non_optimal(seed) # true: non-optimal, false: optimal

        oracle = [is_replay_pass, is_non_optimal, diff_iou_value, lane_id] # [True, True] is violation
        logger.info(f"optimal oracle: {oracle}")
        return oracle
