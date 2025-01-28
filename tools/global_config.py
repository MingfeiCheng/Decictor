from omegaconf import DictConfig
import os

from loguru import logger

class GlobalConfig:

    # root paths
    apollo_root: str = ""
    project_root: str = ""
    output_root: str = ""

    # configs - path
    map_dir: str = ""
    seed_dir: str = ""
    save_dir: str = ""
    container_record_dir: str = ""

    # configs
    hd_map: str = ""

    # Fuzzer
    debug: bool = False

    @staticmethod
    def print():
        logger.info("Global Config:")
        logger.info(f"apollo_root: {GlobalConfig.apollo_root}")
        logger.info(f"project_root: {GlobalConfig.project_root}")
        logger.info(f"output_root: {GlobalConfig.output_root}")
        logger.info(f"map_dir: {GlobalConfig.map_dir}")
        logger.info(f"seed_dir: {GlobalConfig.seed_dir}")
        logger.info(f"save_dir: {GlobalConfig.save_dir}")
        logger.info(f"container_record_dir: {GlobalConfig.container_record_dir}")
        logger.info(f"hd_map: {GlobalConfig.hd_map}")
        logger.info(f"debug: {GlobalConfig.debug}")


# class GlobalConfig:
#
#     __instance = None
#
#     def __init__(self, cfg: DictConfig):
#         self.cfg = cfg
#         self.attributes = dict()
#         self.map_parser = None
#
#         GlobalConfig.__instance = self
#
#     def set_map(self, map_parser):
#         self.map_parser = map_parser
#
#     def update_attribute(self, dk, dv):
#         self.attributes[dk] = dv
#
#     @staticmethod
#     def get_instance() -> 'GlobalConfig':
#         return GlobalConfig.__instance