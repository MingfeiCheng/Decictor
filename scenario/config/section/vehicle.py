from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, TypeVar

from scenario.config.common import MotionWaypoint
from scenario.config.lib.agent_type import AgentType
from scenario.config.section.basic import BasicAgent, BasicSection

AgentTypeClass = TypeVar("AgentTypeClass", bound=AgentType)

@dataclass
class VDAgent(BasicAgent):

    def __init__(self,
                 id: int,
                 mutable: bool,
                 route: List[MotionWaypoint],
                 agent_type: AgentTypeClass,
                 origin_trigger: float,
                 noise_trigger: float):
        category: str = 'vehicle'
        super().__init__(id, mutable, route, agent_type, category, origin_trigger, noise_trigger)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'VDAgent':
        route_js = json_node['route']
        route = list()
        for i, r_js in enumerate(route_js):
            route.append(MotionWaypoint.from_json(r_js))
        json_node['route'] = route
        json_node['agent_type'] = AgentType.from_json(json_node['agent_type'])
        if 'category' in json_node.keys():
            del  json_node['category']
        return cls(**json_node)


@dataclass
class VDSection(BasicSection):

    def __init__(self, agents: List[VDAgent]):
        super().__init__(agents)
        self.id_base = 10000

    @classmethod
    def from_json(cls, json_node: Dict) -> 'VDSection':
        agents_js = json_node['agents']
        agents = list()
        for a_i, a_js in enumerate(agents_js):
            agents.append(VDAgent.from_json(a_js))
        create_json = dict()
        create_json['agents'] = agents
        return cls(**create_json)
