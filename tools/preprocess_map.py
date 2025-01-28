import sys
import time
import os

from loguru import logger

sys.path.append('/data/d/mingfeicheng/Decictor')

from apollo.map_parser import MapParser

def preprocess_apollo_map(map_name: str, apollo_dir: str, map_dir: str):
    """
    Preprocess Apollo map
    """
    logger.info(f'Preprocessing Apollo map: {map_name}')
    start_time = time.time()
    map_instance = MapParser()
    map_instance.parse_from_source(os.path.join(apollo_dir, map_name, 'base_map.bin'))
    map_instance.export(os.path.join(map_dir, map_name))
    logger.info(f'Preprocessing Apollo map: {map_name} finished in {time.time() - start_time:.2f}s')

if __name__ == '__main__':
    project_root = "/data/d/mingfeicheng/Decictor"

    preprocess_apollo_map('sunnyvale_loop', f'{project_root}/data', f'{project_root}/data')
    preprocess_apollo_map('sunnyvale_big_loop', f'{project_root}/data', f'{project_root}/data')