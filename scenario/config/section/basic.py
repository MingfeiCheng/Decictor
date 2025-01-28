import math
import random

from shapely.geometry import Polygon
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, TypeVar

from scenario.config.common import Waypoint
from scenario.config.lib.agent_type import AgentType

WaypointClass = TypeVar("WaypointClass", bound=Waypoint)
AgentTypeClass = TypeVar("AgentTypeClass", bound=AgentType)

@dataclass
class BasicAgent:
    # agent info
    id: int
    mutable: bool # whether mutable True -> mutable False -> fixed
    route: List[WaypointClass]
    # obj info
    agent_type: AgentTypeClass
    category: str
    origin_trigger: float
    noise_trigger: float

    def __init__(self,
                 id: int,
                 mutable: bool,
                 route: List[WaypointClass],
                 agent_type: AgentTypeClass,
                 category: str,
                 origin_trigger: float,
                 noise_trigger: float = 0.0
                 ):
        self.id = id
        self.mutable = mutable
        self.route = route
        self.agent_type = agent_type
        self.category = category
        self.origin_trigger = origin_trigger
        self.noise_trigger = noise_trigger

    @property
    def trigger(self):
        return self.origin_trigger + self.noise_trigger

    def get_initial_polygon(self) -> Polygon:
        half_w = self.agent_type.width / 2.0
        if self.category == 'ads':
            assert self.agent_type.back2center != 0.0
            front_l = self.agent_type.length - self.agent_type.back2center
            back_l = -1 * self.agent_type.back2center
        else:
            front_l = self.agent_type.length / 2.0
            back_l = -1 * self.agent_type.length / 2.0
        location = self.route[0]
        sin_h = math.sin(location.heading)
        cos_h = math.cos(location.heading)
        vectors = [(front_l * cos_h - half_w * sin_h,
                    front_l * sin_h + half_w * cos_h),
                   (back_l * cos_h - half_w * sin_h,
                    back_l * sin_h + half_w * cos_h),
                   (back_l * cos_h + half_w * sin_h,
                    back_l * sin_h - half_w * cos_h),
                   (front_l * cos_h + half_w * sin_h,
                    front_l * sin_h - half_w * cos_h)]
        points = []
        for x, y in vectors:
            points.append([location.x + x, location.y + y])
        start_point = points[0]
        points.append(start_point)
        return Polygon(points)

    def get_initial_waypoint(self) -> WaypointClass:
        return self.route[0]

    def get_destination_waypoint(self) -> WaypointClass:
        return self.route[-1]

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'AgentClass':
        route_js = json_node['route']
        route = list()
        for i, r_js in enumerate(route_js):
            route.append(Waypoint.from_json(r_js))
        json_node['agent_type'] = AgentType.from_json(json_node['agent_type'])
        return cls(**json_node)

AgentClass = TypeVar("AgentClass", bound=BasicAgent)

@dataclass
class BasicSection:

    agents: List[AgentClass]

    __ids: List[int]
    __fixed_ids: List[int]
    __mutant_ids: List[int]

    id_base: int

    def __init__(self, agents: List[AgentClass]):
        self.agents = agents
        self.id_base = 0
        self.__ids = list()
        self.__fixed_ids = list()
        self.__mutant_ids = list()

        for item in agents:
            self.__ids.append(item.id)
            if item.mutable:
                self.__mutant_ids.append(item.id)
            else:
                self.__fixed_ids.append(item.id)

    @property
    def mutant_ids(self):
        return self.__mutant_ids

    @property
    def ids(self):
        return self.__ids

    def get_agent(self, idx) -> Optional[AgentClass]:
        for agent_index, _agent in enumerate(self.agents):
            if _agent.id == idx:
                return _agent
        return None

    def get_new_id(self) -> int:
        new_id = random.randint(0, 10000)
        while self.id_base + new_id in self.ids:
            new_id += 1
            new_id = new_id % 10000
        return self.id_base + new_id

    def add_agent(self, agent: AgentClass):
        assert agent.id not in self.__ids

        self.agents.append(agent)
        self.__ids.append(agent.id)
        if agent.mutable:
            self.__mutant_ids.append(agent.id)
        else:
            self.__fixed_ids.append(agent.id)

    def remove_agent(self, idx: int) -> bool:

        if idx not in self.__ids:
            return False

        target_agent = None
        for item in self.agents:
            if item.id == idx:
                target_agent = item
                break

        if target_agent is None or (not target_agent.mutable):
            return False

        self.agents.remove(target_agent)
        self.__ids.remove(idx)
        self.__mutant_ids.remove(idx)

        return True

    def update_agent(self, idx: int, agent: AgentClass) -> bool:

        if idx not in self.__ids:
            return False

        for agent_index, _agent in enumerate(self.agents):
            if _agent.id == idx:
                self.agents[agent_index] = agent
                break

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'SectionClass':
        agents_js = json_node['agents']
        agents = list()
        for a_i, a_js in enumerate(agents_js):
            agents.append(BasicAgent.from_json(a_js))
        json_node['agents'] = agents
        return cls(**json_node)

SectionClass = TypeVar("SectionClass", bound=BasicSection)