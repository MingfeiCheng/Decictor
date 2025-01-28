from dataclasses import dataclass
from typing import List, Dict

from scenario.config.common import Waypoint
from scenario.config.lib.agent_type import Apollo
from scenario.config.section.basic import BasicAgent, BasicSection

@dataclass
class ADSAgent(BasicAgent):

    def __init__(self,
                 id: int,
                 route: List[Waypoint],
                 origin_trigger: float,
                 noise_trigger: float):
        mutable = False
        agent_type = Apollo()
        category: str = 'ads'
        super().__init__(id, mutable, route, agent_type, category, origin_trigger, noise_trigger)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'ADSAgent':
        route_js = json_node['route']
        route = list()
        for i, r_js in enumerate(route_js):
            route.append(Waypoint.from_json(r_js))
        json_node['route'] = route
        if 'mutable' in json_node.keys():
            del json_node['mutable']
        if 'agent_type' in json_node.keys():
            del json_node['agent_type']
        if 'category' in json_node.keys():
            del json_node['category']
        return cls(**json_node)

@dataclass
class ADSection(BasicSection):

    def __init__(self, agents: List[ADSAgent]):
        super().__init__(agents)
        self.id_base = 0

    @classmethod
    def from_json(cls, json_node: Dict) -> 'ADSection':
        agents_js = json_node['agents']
        agents = list()
        for a_i, a_js in enumerate(agents_js):
            agents.append(ADSAgent.from_json(a_js))
        create_json = dict()
        create_json['agents'] = agents
        return cls(**create_json)
