import os
import sys
import hydra

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from tools.global_config import GlobalConfig
from tools.config_utils import load_entry_point
from apollo.map_parser import MapParser

@hydra.main(config_path='config', config_name='main', version_base=None)
def main(cfg: DictConfig):
    # setup logger
    level = cfg.log_level
    logger.configure(handlers=[{"sink": sys.stderr, "level": level}])

    # set global config
    GlobalConfig.apollo_root = cfg.common.apollo_root
    GlobalConfig.project_root = cfg.common.project_root
    GlobalConfig.output_root = cfg.common.output_root

    # set parameters
    save_name = os.path.join(cfg.seed_name, cfg.fuzzer.name, cfg.run_name)
    GlobalConfig.map_dir = os.path.join(GlobalConfig.project_root, 'data/maps')
    GlobalConfig.seed_dir = os.path.join(GlobalConfig.project_root, f'data/seeds/{cfg.seed_name}')
    GlobalConfig.save_dir = os.path.join(GlobalConfig.output_root, save_name)
    GlobalConfig.container_record_dir = os.path.join(cfg.common.container_record_default_folder, save_name)
    GlobalConfig.hd_map = cfg.map_name

    # others
    GlobalConfig.debug = cfg.debug

    # show cfgs
    GlobalConfig.print()

    # save folder
    save_root = GlobalConfig.save_dir
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    logger_file = os.path.join(save_root, 'run.log')
    if os.path.exists(logger_file):
        os.remove(logger_file)
    _ = logger.add(logger_file, level=level)

    OmegaConf.save(config=cfg, f=os.path.join(save_root, 'run_config.yaml'))
    logger.info('Save result and log to {}', save_root)

    # load map
    map_parser = MapParser()
    map_parser.load_from_file(
        os.path.join(GlobalConfig.map_dir, cfg.map_name)
    ) # abs path for apollo map

    # direct to specific method, such as mr, avfuzz...
    fuzz_cfg = cfg.fuzzer
    runner_cfg = cfg.common
    fuzz_entry_point = fuzz_cfg.entry_point
    fuzz_class = load_entry_point(fuzz_entry_point)

    fuzz_instance = fuzz_class(fuzz_cfg, runner_cfg)
    fuzz_instance.run()


if __name__ == '__main__':
    main()
    logger.info('DONE Fuzzing!')