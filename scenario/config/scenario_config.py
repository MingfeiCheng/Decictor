from dataclasses import dataclass, asdict
from typing import Dict

from scenario.config.section.ads import ADSection
from scenario.config.section.walker import WASection
from scenario.config.section.static import STSection
from scenario.config.section.vehicle import VDSection
from scenario.config.section.region import RESection
from scenario.config.section.traffic_light import TLSection

@dataclass
class ScenarioConfig:

    id: str
    egos: ADSection
    walkers: WASection
    statics: STSection
    vehicles: VDSection
    region: RESection
    traffic_light: TLSection

    def get_npc_ids(self):
        return self.walkers.ids + self.statics.ids + self.vehicles.ids

    def json_data(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, json_node: Dict) -> 'ScenarioConfig':
        json_node['egos'] = ADSection.from_json(json_node['egos'])
        json_node['walkers'] = WASection.from_json(json_node['walkers'])
        json_node['statics'] = STSection.from_json(json_node['statics'])
        json_node['vehicles'] = VDSection.from_json(json_node['vehicles'])
        json_node['region'] = RESection.from_json(json_node['region'])
        json_node['traffic_light'] = TLSection.from_json(json_node['traffic_light'])

        return cls(**json_node)