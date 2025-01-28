import time
import math
import numpy as np

from omegaconf import DictConfig
from threading import Thread

from apollo.apollo_wrapper import ApolloWrapper
from apollo.cyber_bridge import Channel, Topics
from apollo.utils import to_Point3D

from apollo_modules.modules.common.proto.header_pb2 import Header
from apollo_modules.modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacles

from scenario.manager.vehicle_manager import VehicleManager
from scenario.manager.static_manager import StaticManager
from scenario.manager.traffic_control_manager import TrafficControlManager
from scenario.manager.traffic_agent.common import AgentState

class ApolloRunner:

    apollo_wrapper: ApolloWrapper
    vm: VehicleManager
    sm: StaticManager
    tm: TrafficControlManager

    spinning: bool
    t_run: Thread

    PERCEPTION_FREQUENCY = 25.0 # 10.0 # 25.0
    TIME_FREQUENCY = 25.0

    def __init__(self,
                 cfg: DictConfig,
                 apollo_wrapper: ApolloWrapper,
                 vehicle_manager: VehicleManager,
                 static_manager: StaticManager,
                 traffic_light_manager: TrafficControlManager
                 ) -> None:
        """
        Constructor
        """
        self.apollo_wrapper = apollo_wrapper
        self.vm = vehicle_manager
        self.sm = static_manager
        self.tm = traffic_light_manager

        self.spinning = False

        dest_waypoint = apollo_wrapper.get_destination()
        self._apollo_destination = np.array([dest_waypoint.x, dest_waypoint.y])  # np.array([dest_wp.x, dest_wp.y])
        self._prev_location = None

        self.curr_time = 0.0
        self.threshold_dest = cfg.threshold_dest

    @property
    def time(self):
        return self.curr_time

    def broadcast(self, channel: Channel, data: bytes):
        """
        Sends data to specified channel of every instance

        :param Channel channel: cyberRT channel to send data to
        :param bytes data: data to be sent
        """
        # with self.lock:
        self.apollo_wrapper.container.bridge.publish(channel, data)

    def _get_ego_state(self, loc) -> AgentState:
        linear_velocity = to_Point3D(loc.pose.linear_velocity)
        x, y, z = linear_velocity.x, linear_velocity.y, linear_velocity.z
        speed = round(math.sqrt(x ** 2 + y ** 2), 2)
        ego_state = AgentState(
            self.apollo_wrapper.nid,
            'ego',
            x=0.0 if math.isnan(loc.pose.position.x) else loc.pose.position.x,
            y=0.0 if math.isnan(loc.pose.position.y) else loc.pose.position.y,
            heading=loc.pose.heading,
            speed=speed,
            acceleration=0.0,
            steering=0.0,
            length=4.933,
            width=2.11,
            height=1.48,
            back2center=1.043
        )
        return ego_state

    def _spin_run(self):
        self.curr_time = 0.0
        last_time = self.curr_time
        header_sequence_num = 0
        continuous_stuck_count = 0
        apollo_reach_dest_time = 0.0
        stuck_time = 0.0
        last_obstacles_polygons = list()
        while self.spinning:
            # 1. check send route
            self.apollo_wrapper.send_route(self.curr_time)
            # 2. check localization
            loc = self.apollo_wrapper.localization
            if loc and loc.header.module_name == 'SimControl':
                ego_state = self._get_ego_state(loc)
                ego_obs_poly = ego_state.get_polygon()

                # 3. check.
                # 3-1. check collision
                min_dist = 1000
                for obs_ in last_obstacles_polygons:
                    dist_ = obs_.distance(ego_obs_poly)
                    if dist_ < min_dist:
                        min_dist = dist_
                self.apollo_wrapper.set_min_distance(min_dist)
                if self.apollo_wrapper.get_min_distance() <= 0.0:
                    self.apollo_wrapper.set_violations('collision')
                    self.spinning = False
                # 3-2. check timeout
                if self.curr_time > self.apollo_wrapper.time_limit:
                    self.apollo_wrapper.set_violations('timeout')
                    self.spinning = False
                # 3-3. check stuck
                delta_time = float(np.clip(self.curr_time - last_time, 0, None))
                loc_np = np.array([loc.pose.position.x, loc.pose.position.y])
                if self._prev_location is None:
                    self._prev_location = loc_np
                if np.linalg.norm(loc_np - self._prev_location) <= 0.01:
                    continuous_stuck_count += 1
                else:
                    continuous_stuck_count = 0
                if continuous_stuck_count > 0:
                    stuck_time += delta_time
                else:
                    stuck_time = 0.0
                self._prev_location = loc_np
                if stuck_time > self.apollo_wrapper.stuck_time_limit:
                    self.apollo_wrapper.set_violations('stuck')
                    self.spinning = False

                # 3-4. check destination
                apollo_dist2dest = np.linalg.norm(loc_np - self._apollo_destination)
                self.apollo_wrapper.set_dist2dest(apollo_dist2dest)
                if apollo_dist2dest <= self.threshold_dest:
                    apollo_reach_dest_time += delta_time
                else:
                    apollo_reach_dest_time = 0.0
                if apollo_reach_dest_time > 10.0:
                    self.spinning = False

                # 4. next step
                static_perception, static_polygons, static_states = self.sm.run_step(self.curr_time)
                vehicle_perception, vehicle_polygons, vehicle_states = self.vm.run_step(self.curr_time, ego_state, static_states)
                last_obstacles_polygons = static_polygons + vehicle_polygons

                perception_obs = static_perception + vehicle_perception
                header = Header(
                    timestamp_sec=time.time(),
                    module_name='MAGGIE',
                    sequence_num=header_sequence_num
                )
                bag = PerceptionObstacles(
                    header=header,
                    perception_obstacle=perception_obs,
                )
                tld = self.tm.get_traffic_lights(self.curr_time)
                self.broadcast(Topics.TrafficLight, tld.SerializeToString())
                self.broadcast(Topics.Obstacles, bag.SerializeToString())

                header_sequence_num += 1
                last_time = self.curr_time
                self.curr_time += 1 / self.PERCEPTION_FREQUENCY
                time.sleep(1 / self.PERCEPTION_FREQUENCY)

    def spin(self):
        """
        Starts to forward localization
        """
        if self.spinning:
            return

        self.spinning = True
        # self.t_run = Thread(target=self._spin_run)
        self.t_run = Thread(target=self._spin_run)
        self.t_run.start()

    def stop(self):
        """
        Stops forwarding localization
        """
        if not self.spinning:
            return
        self.spinning = False
        self.t_run.join()
        self.apollo_wrapper.container.stop_bridge()