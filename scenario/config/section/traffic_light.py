from dataclasses import dataclass, asdict
from typing import Dict, Optional

@dataclass
class TLSection:

    initial: Dict[str, str]
    duration_g: float
    duration_y: float
    duration_r: float

    def __init__(self, initial: Dict[str, str], duration_g: float, duration_y: float, duration_r: float):
        self.initial = initial
        self.duration_g = duration_g
        self.duration_y = duration_y
        self.duration_r = duration_r

        self.curr_state = self.initial

    def calculate_transition_yellow(self) -> Dict[str, str]:
        for k in self.curr_state:  # all lights
            if self.curr_state[k] == 'GREEN':
                self.curr_state[k] = 'YELLOW'
        return self.curr_state

    def calculate_transition_red(self) -> Dict[str, str]:
        # TODO: Optimize the transition, like generator
        for k in self.curr_state:  # all lights
            if self.curr_state[k] == 'GREEN' or self.curr_state[k] == 'YELLOW':
                self.curr_state[k] = 'RED'
            else:
                self.curr_state[k] = 'GREEN'
        return self.curr_state

    def get_force_green_config(self) -> Dict[str, str]:
        result = dict()
        for k in self.initial:
            result[k] = 'GREEN'
        return result

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'TLSection':
        return cls(**json_node)