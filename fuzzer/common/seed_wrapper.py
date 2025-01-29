import json
import os

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict

from scenario.config.scenario_config import ScenarioConfig
from fuzzer.common.record_data import RecordData

@dataclass
class SeedWrapper:

    id: Any
    prev_id: Any

    scenario: ScenarioConfig

    result: Optional[Dict] # testing result (collision, not reach destination)
    record: Optional[RecordData]

    fitness: Optional[float]
    oracle: Optional[List] # custom oracle (i.e., no optimal scenarios)


    def __init__(self,
                 id: Any,
                 scenario: ScenarioConfig,
                 save_root: Optional[str],
                 prev_id: Any = None):
        # id for comment and file name
        self.id = id
        self.prev_id = prev_id
        self.save_root = save_root

        # scenario config -> need to be saved
        self.scenario = scenario

        # results - simulator testing
        self.result = dict()
        self.record = RecordData() # record data (json) not apollo -> need to be saved

        # oracle & fitness
        self.fitness = 0.0  # default is np.inf, should manual setting according to the optimization in fuzzing
        self.oracle = list()

        self.record_name = f'record_{self.id}' # this is done at basic_runner
        self.scenario_name = f'scenario_{self.id}'
        self.result_name = f'result_{self.id}'

    @property
    def record_item(self):
        return [self.id, self.oracle, self.fitness, self.result, self.scenario_name, self.result_name, self.record_name, self.save_root]

    @property
    def record_columns(self):
        return ['id', 'oracle', 'fitness', 'result', 'scenario_name', 'result_name', 'record_name', 'save_root']

    @property
    def apollo_record_folder(self):
        apollo_record_folder = os.path.join(self.save_root, 'record_apollo')
        if not os.path.exists(apollo_record_folder):
            os.makedirs(apollo_record_folder)
        return apollo_record_folder

    @property
    def json_record_folder(self):
        json_record_folder = os.path.join(self.save_root, 'record')
        if not os.path.exists(json_record_folder):
            os.makedirs(json_record_folder)
        return json_record_folder

    @property
    def scenario_folder(self):
        scenario_folder = os.path.join(self.save_root, 'scenario')
        if not os.path.exists(scenario_folder):
            os.makedirs(scenario_folder)
        return scenario_folder

    @property
    def result_folder(self):
        result_folder = os.path.join(self.save_root, 'result')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        return result_folder

    @property
    def apollo_record_path(self) -> str:
        return os.path.join(self.apollo_record_folder, self.record_name)

    @property
    def json_record_path(self) -> str:
        return os.path.join(self.json_record_folder, f"{self.record_name}.json")

    @property
    def scenario_path(self) -> str:
        return os.path.join(self.scenario_folder, f"{self.scenario_name}.json")

    @property
    def result_path(self) -> str:
        return os.path.join(self.result_folder, f"{self.result_name}.json")

    def update_id(self, idx: Any):
        self.id = idx
        self.scenario.id = idx
        self.record_name = f'record_{self.id}'  # this is done at basic_runner
        self.scenario_name = f'scenario_{self.id}'
        self.result_name = f'result_{self.id}'

    def update_fitness(self, fitness: float):
        self.fitness = fitness

    def update_oracle(self, oracle: List):
        self.oracle = oracle

    def update_result(self, result: Dict):
        self.result = result

    def update_record(self):
        self.record.load_from_json(self.json_record_path)

    def static_number(self):
        return len(self.scenario.statics.agents)

    def vehicle_number(self):
        return len(self.scenario.vehicles.agents)

    def walker_number(self):
        return len(self.scenario.walkers.agents)

    def save(self):
        """
        Save scenario config
        """
        with open(self.scenario_path, 'w') as f:
            json.dump(self.scenario.json_data(), f, indent=4)

        with open(self.result_path, 'w') as f:
            json.dump(self.json_data(), f, indent=4)

    def json_data(self):
        data = {
            'id': self.id,
            'prev_id': self.prev_id,
            'result': self.result,
            'oracle': self.oracle,
            'fitness': self.fitness,
            'scenario': self.scenario_name,
            'record': self.record_name,
            'save_root': self.save_root
        }
        return data

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        json_data["save_root"] = Path(json_file).parent.parent
        scenario_json_file = os.path.join(json_data['save_root'], 'scenario', f"{json_data['scenario']}.json")
        with open(scenario_json_file, 'r') as f:
            scenario_json_data = json.load(f)
        scenario = ScenarioConfig.from_json(scenario_json_data)
        json_data['scenario'] = scenario

        seed_create_json = {
            "id": json_data['id'],
            "scenario": scenario,
            "save_root": json_data['save_root'],
            "prev_id": json_data['prev_id']
        }
        seed = cls(**seed_create_json)

        seed.fitness = json_data['fitness']
        seed.oracle = json_data['oracle']
        seed.result = json_data['result']

        seed.record = RecordData()
        seed.record.load_from_json(os.path.join(json_data['save_root'], 'record', f"{json_data['record']}.json"))

        return seed

