import json
import numpy as np

from typing import Dict, List

class EgoData:
    def __init__(self):
        self.timestamps = None
        self.trace_pts = None

        self.attributes = None

        self.lane_grid = None
        self.lane_occupancy = None

    def load_from_json_node(self, json_node: Dict):
        self.timestamps = list()
        self.trace_pts = list()
        self.attributes = list()

        start_time = json_node[0]['timestamp']
        for frame_info in json_node:
            timestamp = float(np.clip((frame_info['timestamp'] - start_time) * 1e-9, 0.0, None))
            if len(self.timestamps) > 0 and timestamp <= self.timestamps[-1]:
                continue
            self.timestamps.append(timestamp) # second
            self.trace_pts.append([frame_info['position'][0], frame_info['position'][1]])
            # self.trace_polygon.append(Polygon(frame_info['polygon']))
            self.attributes.append([frame_info['position'][0],  # x
                                    frame_info['position'][1],  # y
                                    frame_info['position'][3],  # heading
                                    frame_info['speed'],
                                    frame_info['acceleration']])
        # self.trace = LineString(self.trace_pts)

class EnvData:
    def __init__(self):
        self.timestamps = None
        self.participants = None

    def load_from_json_node(self, json_node: List):
        self.timestamps = list()
        self.participants = list()
        if len(json_node) > 0:
            start_time = json_node[0]['timestamp']
            for frame in json_node:
                timestamp = float(np.clip((frame['timestamp'] - start_time) * 1e-9, 0.0, None))
                if len(self.timestamps) > 0 and timestamp <= self.timestamps[-1]:
                    continue
                self.timestamps.append(timestamp)
                self.participants.append(frame['obstacles'])

class RecordData:

    def __init__(self):

        self.ego = EgoData()
        self.env = EnvData()

    def load_from_json(self, recording_file: str):
        self.ego = EgoData()
        self.env = EnvData()
        
        with open(recording_file, 'r') as f:
            data = json.load(f)

        self.ego.load_from_json_node(data['localization'])
        self.env.load_from_json_node(data['perception'])

    def load_from_json_data(self, json_data: dict):
        self.ego = EgoData()
        self.env = EnvData()

        self.ego.load_from_json_node(json_data['localization'])
        self.env.load_from_json_node(json_data['perception'])