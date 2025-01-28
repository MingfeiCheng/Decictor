import copy
import math
import os.path
import random
import numpy as np

from typing import List
from shapely.geometry import Point
from collections import deque

from scenario.config.section.vehicle import VDAgent
from scenario.config.common import MotionWaypoint, PositionUnit
from scenario.manager.traffic_agent.common import AgentState, KinematicBicycleModel, PIDLateralController, PIDLongitudinalController
from apollo.map_parser import MapParser
from tools.global_config import GlobalConfig
from tools.local_logger import get_instance_logger

normalise_angle = lambda angle: math.atan2(math.sin(angle), math.cos(angle))

class WaypointAgent(object):

    MIN_DISTANCE_PERCENTAGE = 0.95

    def __init__(self,
                 agent: VDAgent):
        # for debug log
        if GlobalConfig.debug:
            self.debug = True
        else:
            self.debug = False

        # self.debug = True

        self._vd_agent = agent
        self._ma = MapParser.get_instance()

        radius = 2.0
        max_speed = 25.0
        max_speed_junction = 15.0
        # 1. other parameters
        self._continuous_running = False
        self._ignore_vehicles = False
        self._min_distance = radius * self.MIN_DISTANCE_PERCENTAGE
        self._max_speed = max_speed  # m/s
        self._max_speed_junction = max_speed_junction
        self._max_steering = float(min(0.85, self._vd_agent.agent_type.max_steer))  # 22 degrees
        self._max_acceleration = 6  # 3 m / s^2
        self._collision_vehicle_threshold = 5.0
        self._finish_time = 0.0
        self._finish_buffer = 20.0 # seconds

        # 2. set waypoints
        self._waypoints_queue = deque(maxlen=max(5000, len(agent.route) + 100))
        for i, waypoint in enumerate(agent.route):
            if i == 0:
                continue
            self._waypoints_queue.append(waypoint)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._curr_state = AgentState(agent.id,
                                      'vehicle',
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

        # 3. other modules
        self.kbm = KinematicBicycleModel(self._vd_agent.agent_type.wheelbase)

        args_lateral_dict = {
            'K_P': 1.0,
            'K_D': 0.0,
            'K_I': 0.05,
            'dt': 1 / 25.0}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': 1 / 25.0}

        self.longitudinal_control = PIDLongitudinalController(**args_longitudinal_dict) # todo: add parameters
        self.lateral_control = PIDLateralController(**args_lateral_dict)

        if self.debug:
            save_root = gc.cfg.save_root
            save_folder = os.path.join(save_root, 'debug_agents/vehicle')
            print(save_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            log_file = os.path.join(save_folder, f"instance_{agent.id}.log")

            self.logger = get_instance_logger(f"instance_{agent.id}", log_file)
            self.logger.info("Logger initialized for this instance")

    def get_curr_state(self) -> AgentState:
        return self._curr_state

    def is_finished(self) -> bool:
        if self._finish_time > self._finish_buffer:
            return True
        else:
            return False

    def run_step(self,
                 delta_time: float,
                 obstacles: List[AgentState],
                 running=True) -> AgentState:

        if not running:
            self._curr_state.speed = 0.0
            return self._curr_state

        # 1. if not end after all waypoints, continuous sample next waypoints
        if self._continuous_running and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)

        # 2. if the queue is empty, stop the car
        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            self._curr_state.speed = 0.0
            self._finish_time += delta_time
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

        # 4.1 detect collision TODO: add walker and others. ignore static
        hazard_detected = self._collision_detected_vehicle(self._curr_state, obstacles, delta_time)
        if hazard_detected:
            curr_speed = self._curr_state.speed
            next_acceleration = (0 - curr_speed) / delta_time + 0.1
            # next_acceleration = - abs(self._max_acceleration) - 8.0

        if self.debug:
            self.logger.info(f"delta_time: {delta_time}")
            self.logger.info(f"roue length: {len(self._vd_agent.route)}")
            self.logger.info(f"next_acceleration: {next_acceleration}")
            self.logger.info(f"next_steering: {next_steering}")
            self.logger.info(f"target heading (waypoint): {target_waypoint.heading}")
            self.logger.info(f"current heading: {self._curr_state.heading}")
            self.logger.info(f"target_waypoint: {target_waypoint.lane_id} ({target_waypoint.x}, {target_waypoint.y})")
            self.logger.info(f"current_waypoint: ({self._curr_state.x}, {self._curr_state.y})")
            self.logger.info(f"distance: {((self._curr_state.x - target_waypoint.x) ** 2 + (self._curr_state.y - target_waypoint.y) ** 2) ** 0.5}")
            self.logger.info(f"hazard_detected: {hazard_detected}")
            self.logger.info('=============End=============')

        # 4.2 run model to estimate the location
        if target_waypoint.is_junction:
            next_state = self.kbm.run_step(self._curr_state, next_acceleration, next_steering, delta_time, self._max_speed_junction)
        else:
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

    def _run_control(self, curr_state: AgentState, target_waypoint: MotionWaypoint, delta_time: float):
        target_speed = target_waypoint.speed
        current_speed = curr_state.speed

        # calculate the acceleration
        next_acceleration = self.longitudinal_control.run_step(current_speed, target_speed, delta_time)
        next_acceleration = float(np.clip(next_acceleration, -self._max_acceleration, self._max_acceleration))

        # calculate the steering.
        target_heading = math.atan2(target_waypoint.y - curr_state.y, target_waypoint.x - curr_state.x)
        current_heading = curr_state.heading
        next_steering = self.lateral_control.run_step(current_heading, target_heading, delta_time)
        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if next_steering > curr_state.steering + 0.1:
            next_steering = curr_state.steering + 0.1
        elif next_steering < curr_state.steering - 0.1:
            next_steering = curr_state.steering - 0.1

        if next_steering >= 0:
            next_steering = min(self._max_steering, next_steering)
        else:
            next_steering = max(-self._max_steering, next_steering)

        if self.debug:
            self.logger.info(f"target heading (calculate): {target_heading}")
        return next_acceleration, next_steering

    def _collision_detected_vehicle(self,
                                    curr_state: AgentState,
                                    vehicle_list: List[AgentState],
                                    delta_t: float):
        if self._ignore_vehicles:
            return False

        # Time to come to a stop
        # time_to_stop = curr_state.speed / float(self._max_acceleration)

        # Total distance traveled
        # distance = curr_state.speed * time_to_stop - 0.5 * self._max_acceleration * time_to_stop ** 2
        distance = curr_state.speed * delta_t + 0.5 * curr_state.acceleration * delta_t ** 2
        distance = float(np.clip(distance, 0.0, None))
        brake_distance = self._collision_vehicle_threshold + distance * 1.2 # >= self._collision_vehicle_threshold

        # print(self._vd_agent.route)
        # print(f"curr_state: {curr_state}")
        buffer_polygon = curr_state.get_front_buffer_polygon(brake_distance)

        # Step 1: get current points and future polygons
        curr_bbs = [curr_state.get_polygon()]
        curr_location = Point([curr_state.x, curr_state.y])
        for i, waypoint in enumerate(self._waypoint_buffer):
            waypoint_location = Point([waypoint.x, waypoint.y])
            if waypoint_location.distance(curr_location) > brake_distance:
                break
            tmp_state = copy.deepcopy(curr_state)
            tmp_state.x = waypoint.x
            tmp_state.y = waypoint.y
            tmp_state.heading = waypoint.heading
            curr_bbs.append(tmp_state.get_polygon())

        for vehicle in vehicle_list:
            # self.logger.debug(f'vehicle id: {vehicle.id}, curr id: {curr_state.id}')
            if vehicle.id == curr_state.id:
                continue
            if vehicle.category not in ['vehicle', 'ego', 'ads']:
                continue
            vehicle_polygon = vehicle.get_polygon()
            if buffer_polygon.intersects(vehicle_polygon):
                return True
            for curr_polygon in curr_bbs:
                if curr_polygon.intersects(vehicle_polygon):
                    return True

        return False

    def _compute_next_waypoints(self, k=1):
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)
        for _ in range(k):
            if not self._waypoints_queue:
                break
            last_waypoint: MotionWaypoint = self._waypoints_queue[-1]
            # sample next
            next_waypoints = self._ma.get_next_waypoint(2.0, last_waypoint.lane_id, last_waypoint.s)
            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) >= 1:
                next_waypoint_cfg = random.choice(next_waypoints)
                point, heading = self._ma.get_coordinate_and_heading(next_waypoint_cfg[0], next_waypoint_cfg[1])
                if last_waypoint.speed > 0:
                    next_speed = last_waypoint.speed
                else:
                    next_speed = random.uniform(1.0, 15.0)

                next_waypoint = MotionWaypoint(
                    origin=PositionUnit(
                        lane_id=next_waypoint_cfg[0],
                        s=next_waypoint_cfg[1],
                        l=0.0,
                        x=point.x,
                        y=point.y,
                        z=0.0,
                        heading=heading
                    ),
                    perturb=PositionUnit(
                        None,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ),
                    origin_speed=next_speed,
                    perturb_speed=0.0,
                    is_junction=self._ma.is_junction_lane(next_waypoint_cfg[0])
                )
                self._waypoints_queue.append(next_waypoint)
