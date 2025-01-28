import math
import time

from typing import List, Any, Tuple
from shapely.geometry import Polygon

from scenario.manager.traffic_agent.vehicle.waypoint_agent import WaypointAgent
from scenario.manager.traffic_agent.common import AgentState

from scenario.config.section.vehicle import VDSection

from apollo_modules.modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacle
from apollo_modules.modules.common.proto.geometry_pb2 import Point3D, PointENU
from tools.utils import generate_polygon

class VehicleManager(object):

    def __init__(self, idx: Any, section: VDSection):
        self.id = idx
        self.section = section
        self.category = 'vehicle'
        self.last_time = 0.0

        self.vehicle_controllers = list()
        self.vehicle_driving_times = list()
        self.vehicle_states = list()
        for i, agent in enumerate(self.section.agents):
            self.vehicle_controllers.append(WaypointAgent(agent))
            self.vehicle_driving_times.append(0.0)
            self.vehicle_states.append(self.vehicle_controllers[i].get_curr_state())

    def _get_perception_obstacle(self, state: AgentState) -> Tuple[PerceptionObstacle, Polygon]:
        loc = PointENU(x=state.x, y=state.y)
        position = Point3D(x=loc.x, y=loc.y, z=loc.z)
        velocity = Point3D(x=math.cos(state.heading) * state.speed,
                           y=math.sin(state.heading) * state.speed, z=0.0)

        obs_polygon, polygon_point = generate_polygon(position.x,
                                                      position.y,
                                                      state.heading,
                                                      front_l=state.length/2.0,
                                                      back_l=state.length/2.0,
                                                      width=state.width,
                                                      z=position.z)

        obs = PerceptionObstacle(
            id=state.id,
            position=position,
            theta=state.heading,
            velocity=velocity,
            acceleration=Point3D(x=0, y=0, z=0),
            length=state.length,
            width=state.width,
            height=state.height,
            type=PerceptionObstacle.VEHICLE,
            timestamp=time.time(),
            tracking_time=1.0,
            polygon_point=polygon_point
        )
        return obs, obs_polygon

    def run_step(self,
                 curr_time: float,
                 ego_state: AgentState,
                 obstacles: List[AgentState]) -> Tuple[List[PerceptionObstacle], List[Polygon], List[AgentState]]:

        apollo_perception = list()
        polygons = list()
        ego_states = [ego_state]

        delta_t = curr_time - self.last_time # time interval
        self.last_time = curr_time

        for index, vd in enumerate(self.section.agents):
            if self.vehicle_controllers[index].is_finished():
                continue

            if curr_time > vd.trigger:
                self.vehicle_driving_times[index] += delta_t
                obs_state = self.vehicle_controllers[index].run_step(delta_t, ego_states + obstacles + self.vehicle_states, True)
                self.vehicle_states[index] = obs_state
            else:
                obs_state = self.vehicle_controllers[index].run_step(delta_t, ego_states + obstacles + self.vehicle_states, running=False)
                self.vehicle_states[index] = obs_state

            obs_perception, obs_polygon = self._get_perception_obstacle(obs_state)
            apollo_perception.append(obs_perception)
            polygons.append(obs_polygon)

        return apollo_perception, polygons, self.vehicle_states

    def add_agent(self, agent):
        if len(self.section.agents) > 10:
            return
        self.section.add_agent(agent)
        self.vehicle_controllers.append(WaypointAgent(agent))
        self.vehicle_driving_times.append(0.0)
        self.vehicle_states.append(self.vehicle_controllers[-1].get_curr_state())