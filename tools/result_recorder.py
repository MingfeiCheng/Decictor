import os
import pandas as pd
from datetime import datetime
from typing import List
from loguru import logger

class ResultRecorder:
    """
    Helper class to track violations detected during scenario generation
    """
    results: List

    def __init__(self, save_folder) -> None:
        self.save_folder = save_folder
        self.results = list()

    def add_item(self, one_result: List):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        one_result.append(date_time)

        self.results.append(one_result)
        # self.results.append((f"{item}" for item in one_result))

    def write_file(self, column_names: List):
        column_names.append('timestamp')
        df = pd.DataFrame(columns=column_names)
        for scenario in self.results:
            # logger.debug(f'scenario: {len(scenario)} column: {len(column_names)}')
            df.loc[len(df.index)] = [
                *scenario
            ]
        df.to_csv(os.path.join(self.save_folder, "results.csv"))

    def clear(self):
        """
        Clears all tracked violations
        """
        self.results = list()

    def print(self, column_names: List):
        """
        Helper function to print tracked violations to terminal
        """
        column_names.append('timestamp')
        df = pd.DataFrame(columns=column_names)
        for scenario in self.results:
            df.loc[len(df.index)] = [
                *scenario
            ]
        print(df[column_names])