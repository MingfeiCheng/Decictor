import os
import math
import numpy as np

from typing import List
from shapely.geometry import Point

from collections import deque
from scenario.config.section.walker import WAAgent
from scenario.config.common import WalkerMotionWaypoint
from scenario.manager.traffic_agent.common import AgentState, PIDLateralController, PIDLongitudinalController, KinematicBicycleModel
from tools.global_config import GlobalConfig
from tools.local_logger import get_instance_logger

class CrosswalkFollower:

    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, agent: WAAgent):

        if GlobalConfig.debug:
            self.debug = True
        else:
            self.debug = False

        self._agent = agent
        self._waypoints_queue = deque(maxlen=max(5000, len(agent.route) + 100))
        for i, waypoint in enumerate(agent.route):
            if i == 0:
                continue
            self._waypoints_queue.append(waypoint)
        self._ignore_vehicles = False
        self._buffer_size = 5
        self._finish_time = 0.0
        self._finish_buffer = 20.0  # seconds
        self._max_acceleration = 2.943 # m/s^2
        self._max_speed = 1.4 # m/s
        radius = 0.5
        self._min_distance = radius * self.MIN_DISTANCE_PERCENTAGE
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._curr_state = AgentState(agent.id,
                                      'walker',
                                      agent.route[0].x,
                                      agent.route[0].y,
                                      agent.route[0].heading,
                                      0.0,
                                      0.0,
                                      0.0,
                                      agent.agent_type.length,
                                      agent.agent_type.width,
                                      agent.agent_type.height,
                                      agent.agent_type.back2center)
        if self.debug:
            save_root = gc.cfg.save_root
            save_folder = os.path.join(save_root, 'debug_agents/walker')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            log_file = os.path.join(save_folder, f"instance_{agent.id}.log")

            self.logger = get_instance_logger(f"instance_{agent.id}", log_file)
            self.logger.info("Logger initialized for this instance")

        args_lateral_dict = {
            'K_P': 1.0,
            'K_D': 0.0,
            'K_I': 0.05,
            'dt': 0.04}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': 0.04}

        self.longitudinal_control = PIDLongitudinalController(**args_longitudinal_dict)  # todo: add parameters
        self.lateral_control = PIDLateralController(**args_lateral_dict)

        self.kbm = KinematicBicycleModel(0.1)

    def get_curr_state(self) -> AgentState:
        return self._curr_state

    def run_step(self,
                 delta_time: float,
                 obstacles: List[AgentState],
                 running=True) -> AgentState:

        if not running:
            self._curr_state.speed = 0.0
            return self._curr_state

        # 2. if the queue is empty, stop the car
        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            self._curr_state.speed = 0.0
            return self._curr_state

        # 3. buffering the waypoints
        if not self._waypoint_buffer:
            for _ in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # 4. run with target_waypoint
        if self.debug:
            self.logger.info('=============Start=============')

        target_waypoint = self._waypoint_buffer[0]
        next_acceleration, next_steering = self._run_control(self._curr_state, target_waypoint, delta_time) # current acceleration & steering

        if self.debug:
            self.logger.info(f"delta_time: {delta_time}")
            self.logger.info(f"roue length: {len(self._agent.route)}")
            self.logger.info(f"next_acceleration: {next_acceleration}")
            self.logger.info(f"next_steering: {next_steering}")
            self.logger.info(f"target heading (waypoint): {target_waypoint.heading}")
            self.logger.info(f"current heading: {self._curr_state.heading}")
            self.logger.info(f"target_waypoint: {target_waypoint.lane_id} ({target_waypoint.x}, {target_waypoint.y})")
            self.logger.info(f"current_waypoint: ({self._curr_state.x}, {self._curr_state.y})")
            self.logger.info(f"distance: {((self._curr_state.x - target_waypoint.x) ** 2 + (self._curr_state.y - target_waypoint.y) ** 2) ** 0.5}")
            self.logger.info('=============End=============')

        hazard_detected = self._collision_detected_vehicle(self._curr_state, obstacles)
        if hazard_detected:
            next_acceleration = - abs(self._max_acceleration) - 2.0

        # 4.2 run model to estimate the location
        next_state = self.kbm.run_step(self._curr_state, next_acceleration, next_steering, delta_time, self._max_speed)

        # 5. purge the queue of obsolete waypoints
        max_index = -1
        next_location = Point([next_state.x, next_state.y])
        curr_location = Point([self._curr_state.x, self._curr_state.y])
        for i, waypoint in enumerate(self._waypoint_buffer):
            waypoint_location = Point([waypoint.x, waypoint.y])
            if waypoint_location.distance(next_location) < self._min_distance or waypoint_location.distance(curr_location) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        self._curr_state = next_state
        return self._curr_state

    def _run_control(self, curr_state: AgentState, target_waypoint: WalkerMotionWaypoint, delta_time: float):
        target_speed = target_waypoint.speed
        current_speed = curr_state.speed

        # calculate the acceleration
        next_acceleration = self.longitudinal_control.run_step(current_speed, target_speed, delta_time)
        next_acceleration = float(np.clip(next_acceleration, -self._max_acceleration, self._max_acceleration))

        # calculate the steering.
        target_heading = math.atan2(target_waypoint.y - curr_state.y, target_waypoint.x - curr_state.x)
        current_heading = curr_state.heading
        next_steering = self.lateral_control.run_step(current_heading, target_heading, delta_time)

        if self.debug:
            self.logger.info(f"target heading (calculate): {target_heading}")
        return next_acceleration, next_steering

    def _collision_detected_vehicle(self,
                                    curr_state: AgentState,
                                    vehicle_list: List[AgentState]):

        if self._ignore_vehicles:
            return False

        # Time to come to a stop
        time_to_stop = curr_state.speed / float(self._max_acceleration)

        # Total distance traveled
        distance = curr_state.speed * time_to_stop - 0.5 * self._max_acceleration * time_to_stop ** 2
        brake_distance = 1.5 + distance * 1.2 # >= self._collision_vehicle_threshold

        buffer_polygon = curr_state.get_front_buffer_polygon(brake_distance)

        for vehicle in vehicle_list:
            if vehicle.id == curr_state.id:
                continue
            if vehicle.category not in ['ego']:
                continue

            vehicle_polygon = vehicle.get_polygon()
            if buffer_polygon.intersects(vehicle_polygon):
                return True
        return False