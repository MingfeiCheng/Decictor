from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

@dataclass
class RESection:
    """
    Genetic representation of the ADS instance section
    """
    lanes_ego: List[str]
    lanes_vehicle: List[str]
    lanes_static: List[str]
    lanes_walker: List[str]
    crosswalks: List[str]
    junctions: List[str]

    def __init__(self, lanes_ego, lanes_vehicle, lanes_static, lanes_walker, crosswalks, junctions):
        self.lanes_ego = lanes_ego
        self.lanes_vehicle = lanes_vehicle
        self.lanes_static = lanes_static
        self.lanes_walker = lanes_walker
        self.crosswalks = crosswalks
        self.junctions = junctions

    @property
    def lanes(self):
        return list(set(self.lanes_static + self.lanes_vehicle + self.lanes_ego))

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'RESection':
        return cls(**json_node)
