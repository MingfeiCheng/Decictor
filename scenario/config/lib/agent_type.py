from typing import Dict
from dataclasses import dataclass, asdict

@dataclass
class AgentType:
    type: str = 'none'
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    back2center: float = 0.0
    wheelbase: float = 0.0
    max_steer: float = 0.0

    def json_data(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict):
        return cls(**json_node)

@dataclass
class Apollo(AgentType):
    type: str = 'ads.apollo'
    length: float = 4.933
    width: float = 2.11
    height: float = 1.48
    back2center: float = 1.043


@dataclass
class Pedestrian(AgentType):
    type: str = 'walker.pedestrian'
    length: float = 0.3
    width: float = 0.6
    height: float = 1.75

@dataclass
class TrafficCone(AgentType):
    type: str = 'static.traffic_cone'
    length: float = 0.172 * 2 #0.172036 * 2
    width: float = 0.172 * 2 # 0.172036 * 2
    height: float = 0.292885 * 2

@dataclass
class SmallCar(AgentType):
    type: str = 'vehicle.small_car'
    length: float = 4.933
    width: float = 2.11
    height: float = 1.48
    back2center: float = 0.0
    wheelbase: float = 2.85
    max_steer: float = 0.8

@dataclass
class Bicycle(AgentType):
    type: str = 'vehicle.bicycle'
    length: float = 2.0
    width: float = 0.6
    height: float = 1.8
    back2center: float = 0.0
    wheelbase: float = 1.0
    max_steer: float = 0.9

@dataclass
class TriggerAgent(AgentType):
    type: str = 'static.trigger'
    length: float = 0.1* 2
    width: float = 0.1 * 2
    height: float = 0.1 * 2