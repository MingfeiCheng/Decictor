from dataclasses import dataclass
from typing import List, Dict, TypeVar

from scenario.config.common import WalkerMotionWaypoint
from scenario.config.lib.agent_type import AgentType
from scenario.config.section.basic import BasicAgent, BasicSection

AgentTypeClass = TypeVar("AgentTypeClass", bound=AgentType)

@dataclass
class WAAgent(BasicAgent):
    def __init__(self,
                 id: int,
                 mutable: bool,
                 route: List[WalkerMotionWaypoint],
                 agent_type: AgentTypeClass,
                 origin_trigger: float,
                 noise_trigger: float):
        category: str = 'walker'
        super().__init__(id, mutable, route, agent_type, category, origin_trigger, noise_trigger)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'WAAgent':
        route_js = json_node['route']
        route = list()
        for i, r_js in enumerate(route_js):
            route.append(WalkerMotionWaypoint.from_json(r_js))
        json_node['route'] = route
        json_node['agent_type'] = AgentType.from_json(json_node['agent_type'])
        if 'category' in json_node.keys():
            del json_node['category']
        return cls(**json_node)

@dataclass
class WASection(BasicSection):

    def __init__(self, agents: List[WAAgent]):
        super().__init__(agents)
        self.id_base = 70000

    @classmethod
    def from_json(cls, json_node: Dict) -> 'WASection':
        agents_js = json_node['agents']
        agents = list()
        for a_i, a_js in enumerate(agents_js):
            agents.append(WAAgent.from_json(a_js))
        create_json = dict()
        create_json['agents'] = agents
        return cls(**create_json)