from typing import Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class PositionUnit:

    lane_id: Optional[str]
    s: float
    l: float
    x: float
    y: float
    z: float
    heading: float

    def json_data(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict):
        return cls(**json_node)

@dataclass
class Waypoint:

    origin: PositionUnit
    perturb: PositionUnit

    @property
    def lane_id(self):
        return self.origin.lane_id

    @property
    def s(self):
        return self.origin.s + self.perturb.s

    @property
    def l(self):
        return self.origin.l + self.perturb.l

    @property
    def x(self):
        return self.origin.x + self.perturb.x

    @property
    def y(self):
        return self.origin.y + self.perturb.y

    @property
    def z(self):
        return self.origin.z + self.perturb.z

    @property
    def heading(self):
        return self.origin.heading + self.perturb.heading

    def json_data(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict):
        origin_js = json_node['origin']
        json_node['origin'] = PositionUnit.from_json(origin_js)

        perturb_js = json_node['perturb']
        json_node['perturb'] = PositionUnit.from_json(perturb_js)

        return cls(**json_node)

@dataclass
class LaneBoundaryWaypoint(Waypoint):

    boundary: str = 'left'

    @classmethod
    def from_json(cls, json_node: Dict):
        origin_js = json_node['origin']
        json_node['origin'] = PositionUnit.from_json(origin_js)

        perturb_js = json_node['perturb']
        json_node['perturb'] = PositionUnit.from_json(perturb_js)

        return cls(**json_node)

@dataclass
class MotionWaypoint(Waypoint):

    origin_speed: float
    perturb_speed: float
    is_junction: bool

    @property
    def speed(self):
        return self.origin_speed + self.perturb_speed

    @classmethod
    def from_json(cls, json_node: Dict):
        origin_js = json_node['origin']
        json_node['origin'] = PositionUnit.from_json(origin_js)

        perturb_js = json_node['perturb']
        json_node['perturb'] = PositionUnit.from_json(perturb_js)

        return cls(**json_node)

@dataclass
class WalkerMotionWaypoint(Waypoint):

    origin_speed: float
    perturb_speed: float
    crosswalk_id: Optional[str]

    @property
    def speed(self):
        return self.origin_speed + self.perturb_speed

    @classmethod
    def from_json(cls, json_node: Dict):
        origin_js = json_node['origin']
        json_node['origin'] = PositionUnit.from_json(origin_js)

        perturb_js = json_node['perturb']
        json_node['perturb'] = PositionUnit.from_json(perturb_js)

        return cls(**json_node)