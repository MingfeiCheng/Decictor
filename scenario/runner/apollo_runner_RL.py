import random
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
from scenario.config.common import MotionWaypoint, PositionUnit, Waypoint
from scenario.config.section.vehicle import VDAgent
from scenario.config.lib.agent_type import SmallCar
from shapely.geometry import Point

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

        start_waypoint = apollo_wrapper.route[0]
        dest_waypoint = apollo_wrapper.get_destination()
        self._apollo_destination = np.array([dest_waypoint.x, dest_waypoint.y])  # np.array([dest_wp.x, dest_wp.y])
        self._prev_location = None
        self._curr_location = np.array([start_waypoint.x, start_waypoint.y, start_waypoint.z])
        self._curr_rotation = np.array([0.0, 0.0, 0.0])
        self._curr_heading = 0.0
        self._curr_speed = 0.0
        self._curr_velocity = np.array([0.0, 0.0, 0.0])
        self.tmp_added_pool = []

        # inner
        self.last_obstacles_polygons = list()
        self.ego_obs_poly = None
        self.perception_obstacles = list()

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

        self._curr_location = np.array([loc.pose.position.x, loc.pose.position.y, loc.pose.position.z])
        # self._curr_rotation = np.array([loc.pose.euler_angles.x, loc.pose.euler_angles.y, loc.pose.euler_angles.z])
        if self._prev_location is None:
            self._curr_heading = 0.0
        else:
            self._curr_heading = math.atan2(self._curr_location[1] - self._prev_location[1], self._curr_location[0] - self._prev_location[0])
        self._curr_velocity = np.array([linear_velocity.x, linear_velocity.y, linear_velocity.z])
        self._curr_speed = speed
        return ego_state

    def _spin_run(self):
        self.curr_time = 0.0
        last_time = self.curr_time
        header_sequence_num = 0
        continuous_stuck_count = 0
        apollo_reach_dest_time = 0.0
        stuck_time = 0.0
        self.last_obstacles_polygons = list()
        while self.spinning:
            # 1. check send route
            self.apollo_wrapper.send_route(self.curr_time)
            # 2. check localization
            loc = self.apollo_wrapper.localization
            if loc and loc.header.module_name == 'SimControl':
                ego_state = self._get_ego_state(loc)
                self.ego_obs_poly = ego_state.get_polygon()

                # 3. check.
                # 3-1. check collision
                min_dist = 1000
                for obs_ in self.last_obstacles_polygons:
                    dist_ = obs_.distance(self.ego_obs_poly)
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

                # 4. next step - action
                static_perception, static_polygons, static_states = self.sm.run_step(self.curr_time)
                vehicle_perception, vehicle_polygons, vehicle_states = self.vm.run_step(self.curr_time, ego_state, static_states)
                self.last_obstacles_polygons = static_polygons + vehicle_polygons
                self.perception_obstacles = static_perception + vehicle_perception
                header = Header(
                    timestamp_sec=time.time(),
                    module_name='MAGGIE',
                    sequence_num=header_sequence_num
                )
                bag = PerceptionObstacles(
                    header=header,
                    perception_obstacle=self.perception_obstacles,
                )
                tld = self.tm.get_traffic_lights(self.curr_time)
                self.broadcast(Topics.TrafficLight, tld.SerializeToString())
                self.broadcast(Topics.Obstacles, bag.SerializeToString())

                # 5. get observation & reward
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

    def get_observation(self):
        # ego_x, ego_y, ego_y, rain, fog, wetness, timeofday, signal, rx (rotation), ry, rz, speed

        return np.array([self._curr_location[0], self._curr_location[1], self._curr_location[2], 0, 0, 0, 0, 0, 0.0, 0.0, self._curr_heading, self._curr_speed])

    def apply_action(self, action):
        #
        if self._curr_speed < 1.0 and len(self.tmp_added_pool) > 0:
            return

        print(f"apply action: {action}")
        yaw = self._curr_heading #self._curr_rotation[2]
        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)])
        ego_point = np.array([self._curr_location[0], self._curr_location[1]])
        print(f"ego_point: {ego_point} forward_vec: {forward_vec} right_vec: {right_vec} curr_location: {self._curr_location} curr_speed: {self._curr_speed} curr_rotation: {self._curr_rotation}")
        sample_length = 10
        max_speed = 5.6
        route = list()
        if action == 0: #'Left_Lane_Maintain':
            start_long = 10.0
            start_lateral = -4.0
            end_lateral = -4.0
            lateral_region = [0.0]
        elif action == 1: #'Right_Lane_Maintain':
            start_long = 10.0
            start_lateral = 4.0
            end_lateral = 4.0
            lateral_region = [0.0]
        elif action == 2: #'Current_Lane_Maintain':
            start_long = 20.0
            start_lateral = 0.0
            end_lateral = 0.0
            lateral_region = [0.0]
        elif action == 3: #'Left_Lane_Change':
            start_long = 10.0
            start_lateral = -4.0
            end_lateral = 0.0
            lateral_region = [0.0, -4.0]
        elif action == 4: #'Right_Lane_Change':
            start_long = 10.0
            start_lateral = 4.0
            end_lateral = 0.0
            lateral_region = [0.0, 4.0]
        else:
            raise RuntimeError(f'Invalid action {action}')

        # add start point
        waypoint_speed = random.uniform(0.1, max_speed)
        start_point = ego_point + start_long * forward_vec + start_lateral * right_vec

        print(f"curr_location: {self._curr_location} start_point: {start_point} start_long: {start_long} start_lateral: {start_lateral} end_lateral: {end_lateral} forward_vec: {forward_vec} right_vec: {right_vec}")

        if len(self.tmp_added_pool) > 0:
            # check conflict
            for obs_ in self.last_obstacles_polygons:
                print(obs_.distance(Point(start_point[0], start_point[1])))
                if obs_.distance(Point(start_point[0], start_point[1])) <= 5.0:
                    return

            for existing_point in self.tmp_added_pool:
                if existing_point.distance(Point(start_point[0], start_point[1])) <= 5.0:
                    return

        if np.any(np.isnan(start_point)):
            return

        # exit()
        # sample waypoints
        print('start add vehicle')
        route.append(MotionWaypoint(
            origin=PositionUnit(
                lane_id='none',
                s=0.0,
                l=0.0,
                x=start_point[0],
                y=start_point[1],
                z=0.0,
                heading=yaw
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
            origin_speed=waypoint_speed,
            perturb_speed=0.0,
            is_junction=True
        ))
        for i in range(sample_length):
            waypoint_speed = random.uniform(0.1, max_speed)
            # generate vehicle
            if len(lateral_region) > 1:
                if random.random() < 0.8:
                    mid_lateral = lateral_region[0]
                else:
                    mid_lateral = lateral_region[1]
            else:
                mid_lateral = lateral_region[0]
            mid_point = ego_point + (start_long + (i+1) * 5.0) * forward_vec - mid_lateral * right_vec

            heading = math.atan2(mid_point[1] - route[-1].origin.y, mid_point[0] - route[-1].origin.x)

            # print(
            #     f"curr_location: {self._curr_location} start_point: {start_point} start_long: {start_long} start_lateral: {start_lateral} "
            #     f"end_lateral: {end_lateral} forward_vec: {forward_vec} right_vec: {right_vec} heading: {heading} mid_point: {mid_point}")
            if np.any(np.isnan(mid_point)):
                return

            route.append(MotionWaypoint(
                origin=PositionUnit(
                    lane_id='none',
                    s=0.0,
                    l=0.0,
                    x=mid_point[0],
                    y=mid_point[1],
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
                origin_speed=waypoint_speed,
                perturb_speed=0.0,
                is_junction=True
            ))

        # exit()
        # add end point
        waypoint_speed = 0.0 #random.uniform(0.1, max_speed)
        end_point = ego_point + sample_length * forward_vec + end_lateral * right_vec
        heading = math.atan2(end_point[1] - route[-1].origin.y, end_point[0] - route[-1].origin.x)

        if np.any(np.isnan(end_point)):
            return
        # sample waypoints
        route.append(MotionWaypoint(
            origin=PositionUnit(
                lane_id='none',
                s=0.0,
                l=0.0,
                x=end_point[0],
                y=end_point[1],
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
            origin_speed=waypoint_speed,
            perturb_speed=0.0,
            is_junction=True
        ))

        new_id = self.vm.section.get_new_id()
        vd_trigger = random.uniform(0.0, 5.0)
        agent = VDAgent(new_id, mutable=True, route=route, agent_type=SmallCar(), origin_trigger=vd_trigger, noise_trigger=0.0)
        self.vm.add_agent(agent)
        self.tmp_added_pool.append(Point(start_point[0], start_point[1]))

    def collision_probability(self, dis_tag = False):
        agent_position_x = self._curr_location[0]
        agent_position_y = self._curr_location[1]

        agent_velocity_x = self._curr_velocity[0] if self._curr_velocity[0] != 0 else 0.0001
        agent_velocity_y = self._curr_velocity[1]

        trajectory_ego_k = agent_velocity_y / agent_velocity_x
        trajectory_ego_b = -(agent_position_y / agent_velocity_x) * agent_position_x + agent_position_y

        ego_speed = self._curr_speed if self._curr_speed > 0 else 0.0001
        TTC = 100000
        distance = 100000
        loProC_list, laProC_list = [0], [0]  # probability

        for i in range(0, len(self.perception_obstacles)):
            agent = self.perception_obstacles[i]
            agent_polygon = self.last_obstacles_polygons[i]
            if dis_tag:
                dis = agent_polygon.distance(self.ego_obs_poly) #get_distance(ego, agents[i].transform.position.x, agents[i].transform.position.z)
                distance = dis if dis <= distance else distance

            # print('distance:', get_distance(ego, agents[i].transform.position.x, agents[i].transform.position.z))
            # get agent line
            """
            self._curr_location = np.array([loc.pose.position.x, loc.pose.position.y, loc.pose.position.z])
            self._curr_rotation = np.array([loc.pose.euler_angles.x, loc.pose.euler_angles.y, loc.pose.euler_angles.z])
            self._curr_velocity = np.array([linear_velocity.x, linear_velocity.y, linear_velocity.z])
            self._curr_speed = speed
            """
            agent_position_x = agent.position.x
            agent_position_y = agent.position.y

            agent_velocity_x = agent.velocity.x if agent.velocity.x != 0 else 0.0001
            agent_velocity_y = agent.velocity.y

            trajectory_agent_k = agent_velocity_y / agent_velocity_x
            trajectory_agent_b = -(agent_position_y / agent_velocity_x) * agent_position_x + agent_position_y

            agent_speed = round(math.sqrt(agent_velocity_x ** 2 + agent_velocity_y ** 2), 2)  #agents[i].state.speed if agents[i].state.speed > 0 else 0.0001
            agent_speed = agent_speed if agent_speed > 0 else 0.0001

            # same_lane, ego_ahead, ttc = judge_same_line(ego, agents[i], trajectory_ego_k, trajectory_agent_k)
            # judge same line
            judge = False
            ego_ahead = False
            direction_vector = (self._curr_location[0] - agent.position.x,
                                self._curr_location[1] - agent.position.y)
            distance = math.sqrt(pow(self._curr_location[0] - agent.position.x, 2) + pow(self._curr_location[1] - agent.position.y, 2))
            #get_distance(agent1, agent2.transform.position.x, agent2.transform.position.z)
            k1 = trajectory_ego_k
            k2 = trajectory_agent_k
            if abs(k1 - k2) < 0.6:
                if abs((self._curr_location[1] - agent.position.y) /
                       ((self._curr_location[0] - agent.position.x) if (self._curr_location[0] - agent.position.x) != 0 else 0.01) - (
                               k1 + k2) / 2) < 0.6:
                    judge = True

            if not judge:
                same_lane = judge
                ego_ahead = ego_ahead
                ttc = -1
            else:

                if direction_vector[0] * self._curr_velocity[0] >= 0 and direction_vector[1] * self._curr_velocity[1] >= 0:
                    ego_ahead = True  # Ego ahead of NPC.
                    TTC = distance / (self._curr_speed - agent_speed + 1e-5)
                else:
                    TTC = distance / (self._curr_speed - agent_speed + 1e-5)
                if TTC < 0:
                    TTC = 100000

                same_lane = judge
                ego_ahead = ego_ahead
                ttc = TTC

            ego_deceleration = 6  # probability
            if same_lane:
                # print('Driving on Same Lane, TTC: {}'.format(ttc))
                time_ego = ttc
                time_agent = ttc

                loSD = 100000
                # if agents[i].type == 2:  # type value, 1-EGO, 2-NPC, 3-Pedestrian
                # only support NPC vehicles
                agent_deceleration = 6
                loSD = 1 / 2 * (
                    abs(pow(ego_speed, 2) / ego_deceleration - pow(agent_speed, 2) / agent_deceleration)) + 5
                # else:
                #     agent_deceleration = 1.5
                #     if not ego_ahead:
                #         loSD = 1 / 2 * (
                #                     pow(ego_speed, 2) / ego_deceleration - pow(agent_speed, 2) / agent_deceleration) + 5
                # loProC = calculate_collision_probability(loSD, distance)
                safe_distance = loSD
                current_distance = distance
                collision_probability = None
                if current_distance >= safe_distance:
                    collision_probability = 0
                elif current_distance < safe_distance:
                    collision_probability = (safe_distance - current_distance) / safe_distance
                loProC = collision_probability
                loProC_list.append(loProC)
            else:
                trajectory_agent_k = trajectory_agent_k if trajectory_ego_k - trajectory_agent_k != 0 else trajectory_agent_k + 0.0001

                collision_point_x, collision_point_y = (trajectory_agent_b - trajectory_ego_b) / (
                        trajectory_ego_k - trajectory_agent_k), \
                                                       (
                                                               trajectory_ego_k * trajectory_agent_b - trajectory_agent_k * trajectory_ego_b) / (
                                                               trajectory_ego_k - trajectory_agent_k)

                ego_distance = math.sqrt(pow(self._curr_location[0] - collision_point_x, 2) + pow(self._curr_location[1] - collision_point_y, 2))
                # get_distance(ego, collision_point_x, collision_point_y)
                agent_distance = math.sqrt(pow(agent.position.x - collision_point_x, 2) + pow(agent.position.y - collision_point_y, 2))
                #get_distance(agents[i], collision_point_x, collision_point_y)
                time_ego = ego_distance / ego_speed
                time_agent = agent_distance / agent_speed
                # print('Driving on Different Lane, TTC: {}'.format(time_ego))

                # theta = calculate_angle_tan(trajectory_ego_k, trajectory_agent_k)
                k1 = trajectory_ego_k
                k2 = trajectory_agent_k
                if k1 == k2:
                    k2 = k2 - 0.0001
                tan_theta = abs((k1 - k2) / (1 + k1 * k2))
                theta = np.arctan(tan_theta)

                # print(trajectory_ego_k, trajectory_agent_k, theta)
                laSD = pow(ego_speed * math.sin(theta), 2) / (ego_deceleration * math.sin(theta)) + 5
                # laProC = calculate_collision_probability(laSD, distance)
                safe_distance = laSD
                current_distance = distance
                collision_probability = None
                if current_distance >= safe_distance:
                    collision_probability = 0
                elif current_distance < safe_distance:
                    collision_probability = (safe_distance - current_distance) / safe_distance
                laProC = collision_probability
                laProC_list.append(laProC)

            if abs(time_ego - time_agent) < 1:
                TTC = min(TTC, (time_ego + time_agent) / 2)

        loProC_dt, laProC_dt = max(loProC_list), max(laProC_list)
        proC_dt = max(loProC_dt, laProC_dt) + (1 - max(loProC_dt, laProC_dt)) * min(loProC_dt, laProC_dt)

        return TTC, distance, proC_dt

