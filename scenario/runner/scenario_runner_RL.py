import os
import shutil
import sys
import time
import glob
import json
import math
import traceback

from tqdm import tqdm
from loguru import logger
from datetime import datetime
from omegaconf import DictConfig
from typing import Optional, TypeVar, Dict
from cyber_record.record import Record

from apollo.docker_container import ApolloContainer
from apollo.apollo_wrapper import ApolloWrapper
from apollo.utils import calculate_velocity

from scenario.runner.apollo_runner_RL import ApolloRunner
from scenario.manager.static_manager import StaticManager
from scenario.manager.vehicle_manager import VehicleManager
from scenario.manager.traffic_control_manager import TrafficControlManager

from tools.global_config import GlobalConfig

from fuzzer.common.seed_wrapper import SeedWrapper

SeedWrapperClass = TypeVar("SeedWrapperClass", bound=SeedWrapper)

class ScenarioRunner:
    """
    Executes a scenario based on the specification
    """
    seed: Optional[SeedWrapperClass]
    apollo_container: ApolloContainer
    apollo_wrapper: ApolloWrapper
    vm: VehicleManager
    sm: StaticManager
    tm: TrafficControlManager

    is_initialized: bool
    record_folder: Optional[str]
    __instance = None

    def __init__(self, cfg: DictConfig, policy) -> None:
        """
        Once this instance is created, the initial attributes will not be changed anymore
        cfg: cfg.runner
        """
        self.cfg = cfg
        self.policy = policy
        self.delete_record = cfg.delete_record
        self.force_green = cfg.force_green # traffic light
        self.perception_range = cfg.perception_range
        self.container_record_folder = GlobalConfig.container_record_dir

        self.apollo_container = ApolloContainer(GlobalConfig.hd_map)

        # exit(-1)
        self.is_initialized = False
        self.seed = None

    def run(self, seed: SeedWrapper, save_record=True) -> Optional[SeedWrapper]:
        MAX_TRY = 20
        try_time = 0
        while True:
            try:
                try_time += 1
                if try_time > MAX_TRY:
                    logger.error('Running error (e.g. record) for scenario')
                    sys.exit(-1)
                    # return None
                basic_result = self._run_seed(seed, save_record)
                logger.info('Simulator results: ')
                logger.info(basic_result)

                if try_time < 5:
                    if basic_result['feedback']['destination'] > 500:
                        if basic_result['feedback']['collision'] > 600:
                            # try_time -= 1
                            continue
                seed.update_result(basic_result)
                has_bug = self._convert_record()
                if has_bug:
                    logger.warning('Running error for scenario with record bugs, rerun')
                    self.apollo_container.start_container(restart=True)
                    time.sleep(1)
                    continue
                self.apollo_container.stop_container(1.0)
                seed.update_record()  # update record here
                return seed
            except Exception:
                logger.warning('Running error for scenario with container bugs, rerun')
                logger.warning('{}', traceback.format_exc())
                self.apollo_container.start_container(restart=True)
                time.sleep(1)
                continue

    def __init_scenario(self):
        """
        Initialize the scenario
        """
        # Step1: Start apollo and modules here
        # each scenario, will restart container and start apollo as follows
        self.apollo_container.start_container(restart=True)
        self.apollo_container.clean_apollo_dir()
        self.apollo_container.start_apollo()

        # Step2: Start apollo wrapper here
        scenario = self.seed.scenario
        self.apollo_wrapper = ApolloWrapper(0, self.cfg, self.apollo_container, self.seed.scenario.egos.agents[0].route) # currently only support 1 ADS
        self.apollo_wrapper.initialize()
        # apollo_thread = threading.Thread(target=self.apollo_wrapper.initialize)
        # apollo_thread.start()
        # apollo_thread.join()

        # todo: add perception range
        self.sm = StaticManager('static_manager', scenario.statics)  # @mingfei add
        # wm = WalkerManager('walker_manager', scenario.walkers)
        self.vm = VehicleManager('vehicle_manager', scenario.vehicles)  # @mingfei fix
        self.tm = TrafficControlManager('traffic_light', scenario.traffic_light) # TODO: Check traffic light
        self.tm.set_force_green(self.force_green)

        self.is_initialized = True

    def _run_seed(self, seed: SeedWrapper, save_record=False) -> Optional[Dict]:
        """
        Execute the scenario based on the specification
        """
        self.seed = seed
        logger.info(f'--> Run Seed ID: {self.seed.id}')

        # init scenario and apollo container/modules
        self.__init_scenario()

        if self.seed.scenario is None or not self.is_initialized:
            logger.error('Error: No chromosome or not initialized')
            return None

        if save_record:
            container_record_folder = os.path.join(self.container_record_folder, self.seed.record_name)
            self.apollo_wrapper.container.start_recorder(container_record_folder, 'recording')

        apollo_runner = ApolloRunner(self.cfg, self.apollo_wrapper, self.vm, self.sm, self.tm)
        apollo_runner.spin()

        bar = tqdm()
        # time.sleep(40 / 25.0)  # for latency
        m_start_time = datetime.now()
        # Begin Scenario Cycle

        s = apollo_runner.get_observation()
        step = 0
        monitor_time_start = datetime.now()
        mutation_time_lst = []
        feedback_time_lst = []
        while True:
            curr_time = datetime.now()
            if (curr_time - monitor_time_start).total_seconds() > 300:
                logger.warning('Timeout')
                break

            # TODO: add RL here
            if step % 20 == 0:
                mutation_start_time = datetime.now()
                action = self.policy.choose_action(s)
                apollo_runner.apply_action(action)
                mutation_end_time = datetime.now()
                mutation_time_lst.append((mutation_end_time - mutation_start_time).total_seconds())

                feedback_time_start = datetime.now()
                s_ = apollo_runner.get_observation()
                # calculate reward
                action_reward = 0
                collision_probability = 0
                is_collision = self.apollo_wrapper.get_min_distance() < 0.01
                if is_collision:
                    action_reward = 1
                    collision_probability = 1
                    # episode_done = True
                else:
                    ttc, distance, proC_dt = apollo_runner.collision_probability()
                    collision_probability = round(float(proC_dt), 3)
                    if collision_probability < 0.2:
                        action_reward = -1
                    else:
                        action_reward = collision_probability

                # s_, reward, collision_probability, done, info = calculate_reward(action)
                self.policy.store_transition(s, action, action_reward, s_)
                feedback_time_end = datetime.now()
                feedback_time_lst.append((feedback_time_end - feedback_time_start).total_seconds())

                s = s_
            step += 1

            bar.set_description('--> Scenario time: {}.'.format(round(apollo_runner.time, 2)))
            if not apollo_runner.spinning:
                # Done
                break
            time.sleep(0.05)

        if save_record:
            self.apollo_wrapper.container.stop_recorder()
            time.sleep(1.0)

        # scenario ended
        apollo_runner.stop()
        self.apollo_wrapper.container.bridge.stop()

        self.is_initialized = False
        m_end_time = datetime.now()
        logger.info('--> [Simulation Time] Simulation Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())
        logger.info('--> [Simulation Time] Mutation Spend Time: [=]{}[=]', sum(mutation_time_lst))
        logger.info('--> [Simulation Time] Feedback Spend Time: [=]{}[=]', sum(feedback_time_lst))
        # TODO: check why not return
        # assign result
        basic_result = {
            'violation': list(set(self.apollo_wrapper.get_violations())),
            'feedback': {
                'collision': self.apollo_wrapper.get_min_distance(),
                'destination': self.apollo_wrapper.get_dist2dest()
            }
        }
        return basic_result

    def _convert_record(self) -> bool:
        # @mingfei add -> move to folder
        # {self.container_name}.{record_id}
        # Record in apollo folder: /apollo/records/{self.container_name}.{record_id}
        container_record_folder = os.path.join(self.container_record_folder, self.seed.record_name)
        # copy or move record to local folder
        if os.path.exists(self.seed.apollo_record_path):
            shutil.rmtree(self.seed.apollo_record_path)
        self.apollo_container.copy_record(container_record_folder, self.seed.apollo_record_path, delete=self.delete_record)

        json_record_file = self.seed.json_record_path
        rp = RecordConverter(self.seed.apollo_record_path, json_record_file)
        has_bug = rp.parse()
        return has_bug

class RecordConverter:

    MAX_RETRY = 4  # times
    RETRY_DELAY = 3  # seconds

    def __init__(self, apollo_record_folder: str, record_file: str):
        self.apollo_record_folder = apollo_record_folder
        self.record_file = record_file

        # ads attributes
        self.channel_localization = list()

        # obstacles - env
        # self.channel_perception = dict()
        self.channel_perception = list()
        self.start_time = None

        # status
        self.received_routing = False # finished
        self.estops = set() # finished

        # tmp variables
        self._last_perception = None
        self._prev_pose = None
        self._next_pose = None

    def _parse_channel_localization(self, topic: str, message, t):
        """
        Behavior
        """
        if topic != '/apollo/localization/pose':
            return

        adc_pose = message.pose
        adc_position = adc_pose.position
        adc_heading = adc_pose.heading

        adc_location = [adc_position.x,
                        adc_position.y,
                        adc_position.z,
                        adc_heading]

        # compute speed
        speed = calculate_velocity(message.pose.linear_velocity)

        frame_info = dict()
        frame_info['timestamp'] = t
        frame_info['position'] = adc_location
        frame_info['speed'] = speed
        frame_info['acceleration'] = 0.0
        # compute acceleration
        if self._prev_pose is None and self._next_pose is None:
            self._prev_pose = message
            self.channel_localization.append(frame_info)
            return

        self._next_pose = message
        accel_x = self._next_pose.pose.linear_acceleration.x
        accel_y = self._next_pose.pose.linear_acceleration.y
        accel_z = self._next_pose.pose.linear_acceleration.z

        accel_value = math.sqrt(accel_x ** 2 + accel_y ** 2 + accel_z ** 2)

        prev_velocity = calculate_velocity(self._prev_pose.pose.linear_velocity)
        next_velocity = calculate_velocity(self._next_pose.pose.linear_velocity)
        direction = next_velocity - prev_velocity

        if direction < 0:
            accel = accel_value * -1
        else:
            accel = accel_value
        frame_info['acceleration'] = accel
        self.channel_localization.append(frame_info)
        # update _prev_pose
        self._prev_pose = message

    def _parse_channel_perception(self, topic: str, message, t):
        """
        Perception obstacles
        """
        if topic == '/apollo/perception/obstacles':
            self._last_perception = message
        else:
            return

        if self._last_perception is None:
            # cannot analyze
            return

        frame_info = dict()
        frame_info['timestamp'] = t
        frame_info['obstacles'] = dict()
        for obs in self._last_perception.perception_obstacle:
            # todo: add acceleration?
            obs_frame = dict() # center is x,y and half of length / width
            obs_frame['position'] = [obs.position.x, obs.position.y, obs.position.z, obs.theta]
            obs_frame['shape'] = [obs.length, obs.width, obs.height]
            obs_frame['speed'] = calculate_velocity(obs.velocity)
            obs_frame['polygon'] = [[x.x, x.y] for x in obs.polygon_point]
            frame_info['obstacles'][obs.id] = obs_frame

        self.channel_perception.append(frame_info)

    def _parse_estop(self, topic: str, message):
        """
        Obtain estop flag
        """
        if topic != '/apollo/planning':
            return
        main_decision = message.decision.main_decision
        if main_decision.HasField('estop'):
            self.estops.add(main_decision.estop.reason_code)

    def _parse_module_status(self, topic: str):
        """
        Obtain module status
        """
        if topic == '/apollo/routing_response':
            self.received_routing = True

    def _on_new_message(self, topic, message, t):
        self._parse_channel_localization(topic, message, t)
        self._parse_channel_perception(topic, message, t)
        self._parse_estop(topic, message)
        self._parse_module_status(topic)

    def parse(self) -> bool:
        # todo: add sampling, because too large for some case
        trial = 1
        has_bug = True
        while trial <= self.MAX_RETRY:
            try:
                m_start_time = datetime.now()
                # logger.debug('current record folder: {}, r_prex: {}', self.record_folder, self.r_prex)
                for record_path in sorted(glob.glob(os.path.join(self.apollo_record_folder, 'recording.*')), key=lambda x: int(x[-5:])):
                    record = Record(record_path)
                    # last_t = None
                    for topic, message, t in record.read_messages():
                        # logger.debug('add topic: {}', topic)
                        self._on_new_message(topic, message, t)
                has_bug = self._write_to_pickle()
                m_end_time = datetime.now()
                logger.info('--> Parser Recorder Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())
                return has_bug
            except AttributeError:
                logger.warning('AttributeError {}', traceback.format_exc())
                time.sleep(self.RETRY_DELAY)
                trial += 1
            except FileNotFoundError:
                logger.warning('FileNotFoundError {}', traceback.format_exc())
                time.sleep(self.RETRY_DELAY)
                trial += 1
        return has_bug

    def _write_to_pickle(self) -> bool:
        if len(self.channel_localization) <= 0:
            return True
        record_info = {
            # localization
            'localization': self.channel_localization, # list of frame dict information
            'perception': self.channel_perception, # 'ego' & other participants
            # violations
            'received_routing': self.received_routing,
            'estops': list(self.estops)
        }
        file_path = os.path.join(self.record_file)
        with open(file_path, 'w') as f:
            json.dump(record_info, f, indent=4)
        logger.info(f'--> Save record (json) to {file_path}')
        return False