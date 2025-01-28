import math

from typing import Dict, Optional, Any
from omegaconf import DictConfig
from shapely.geometry import Polygon
from shapely.ops import unary_union

from fuzzer.common.record_data import RecordData
from fuzzer.common.seed_wrapper import SeedWrapper
from tools.utils import generate_polygon

class RegionCalculator:

    def __init__(self, cfg: DictConfig, adv_seed: SeedWrapper):

        self.threshold_ego = cfg.get('threshold_ego', [0.0, 0.0])  # [delta_length, delta_width]
        self.threshold_static = cfg.get('threshold_static', [0.0, 0.0])  # [delta_length, delta_width]
        self.threshold_vehicle = cfg.get('threshold_vehicle', [0.0, 0.0])  # [delta_length, delta_width]

        self.adv_seed = adv_seed
        self.ego_type = self.adv_seed.scenario.egos.agents[0].agent_type

        # regression adv ego function
        self.adv_record = self.adv_seed.record

        # cache for prohibited region
        self.time_list = list()
        self.cache = dict() # t_t+1: obs_id: Polygon or None

        # tmp
        self.global_region = None
        self.frame_region = dict()

    def get_frame_region(self, t: float, next_t: float) -> Polygon:
        frame_key = f"{t}_{next_t}"

        if frame_key in self.frame_region.keys():
            return self.frame_region[frame_key]

        frame_regions = list()
        for npc_key, region in self.cache[frame_key].items():
            frame_regions.append(region)

        self.frame_region[frame_key] = unary_union(frame_regions)
        return self.frame_region[frame_key]

    def get_global_region(self) -> Polygon:
        if self.global_region is None:
            global_list = list()
            for frame_key, participants in self.cache.items():
                frame_regions = list()
                for p_id, p_poly in participants.items():
                    frame_regions.append(p_poly)
                    global_list.append(p_poly)
                self.frame_region[frame_key] = unary_union(frame_regions)
            self.global_region = unary_union(global_list)
        return self.global_region

    def get_global_region_except(self, participant_id: Any) -> Polygon:
        global_list = list()
        for frame_key, participants in self.cache.items():
            for p_id, p_poly in participants.items():
                if str(p_id) == str(participant_id):
                    continue
                if p_poly is None:
                    continue
                global_list.append(p_poly)
        return unary_union(global_list)


    def _prohibited_region_npc(self,
                               obs_id: int,
                               obs_record: Dict) -> Polygon:
        if int(str(obs_id)[0]) == 1:
            # vehicle
            length_threshold = self.threshold_vehicle[0]
            width_threshold = self.threshold_vehicle[1]
        elif int(str(obs_id)[0]) == 4:
            # static
            length_threshold = self.threshold_static[0]
            width_threshold = self.threshold_static[1]
        else:
            raise KeyError(f"Unsupported type: {int(str(obs_id)[0])}")

        polygon, _ = generate_polygon(
            x=obs_record['position'][0],
            y=obs_record['position'][1],
            heading=obs_record['position'][3],
            front_l=obs_record['shape'][0]/2.0 + length_threshold,
            back_l=obs_record['shape'][0]/2.0,
            width=obs_record['shape'][1] + width_threshold
        )

        return polygon

    def _prohibited_region_ego(self,
                               record: RecordData,
                               t: float,
                               next_t: float) -> Polygon:
        timestamps = record.ego.timestamps
        attributes = record.ego.attributes

        # 1. calculate the region from t to t + delta_t
        prohibited_region = list()
        for frame_index, timestamp in enumerate(timestamps):
            if t <= timestamp <= next_t:
                # calculate the polygon
                frame_attribute = attributes[frame_index]
                polygon, _ = generate_polygon(
                    x=frame_attribute[0],
                    y=frame_attribute[1],
                    heading=frame_attribute[2],
                    front_l=self.ego_type.length - self.ego_type.back2center + self.threshold_ego[0],
                    back_l=-1 * self.ego_type.back2center,
                    width=self.ego_type.width + self.threshold_ego[1]
                )
                prohibited_region.append(polygon)
            elif timestamp > next_t:
                break
            else:
                continue

        if t >= timestamps[-1]:
            frame_attribute = attributes[-1]
            polygon, _ = generate_polygon(
                x=frame_attribute[0],
                y=frame_attribute[1],
                heading=frame_attribute[2],
                front_l=self.ego_type.length - self.ego_type.back2center + self.threshold_ego[0],
                back_l=-1 * self.ego_type.back2center,
                width=self.ego_type.width + self.threshold_ego[1]
            )
            prohibited_region.append(polygon)

        union_region = unary_union(prohibited_region)
        return union_region

    def _prohibited_region_participant(self,
                                       participant_id: Any,
                                       record: RecordData,
                                       t: float,
                                       next_t: float) -> Optional[Polygon]:

        timestamps = record.env.timestamps
        participants = record.env.participants

        if len(timestamps) < 0:
            return None

        # 1. calculate the region from t to t + delta_t
        region_list = list()
        for frame_index, timestamp in enumerate(timestamps):
            if t <= timestamp <= next_t:
                frame = participants[frame_index]
                for obs_id, obs_info in frame.items():
                    if str(obs_id) == str(participant_id):
                        obs_region = self._prohibited_region_npc(obs_id, obs_info)
                        region_list.append(obs_region)
            elif timestamp > next_t:
                break
            else:
                continue

        return unary_union(region_list)

    def preprocess(self, source_seed: SeedWrapper, motion_interval: float):
        self.cache = dict()
        self.global_region = None
        self.frame_region = dict()

        start_time = 0.0
        end_time = math.ceil(source_seed.record.ego.timestamps[-1]) + 20
        self.time_list = [start_time + i * motion_interval for i in range(int((end_time - start_time) // motion_interval) + 1)]

        npc_ids = source_seed.scenario.get_npc_ids()
        npc_ids.append('ego')
        for i, t in enumerate(self.time_list[:-1]):
            next_t = self.time_list[i + 1]
            frame_key = f"{t}_{next_t}"
            self.cache[frame_key] = dict()
            for npc_id in npc_ids:
                if npc_id == 'ego':
                    prohibit_region = self._prohibited_region_ego(self.adv_record, t, next_t)
                else:
                    source_record = source_seed.record
                    prohibit_region = self._prohibited_region_participant(npc_id, source_record, t, next_t)
                self.cache[frame_key][npc_id] = prohibit_region # NOTE THAT: None value