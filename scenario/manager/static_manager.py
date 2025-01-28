import time

from apollo_modules.modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacle
from apollo_modules.modules.common.proto.geometry_pb2 import Point3D, PointENU

from typing import List, Any, Tuple
from shapely.geometry import Polygon
from scenario.config.section.static import STSection, STAgent
from scenario.manager.traffic_agent.common import AgentState
from tools.utils import generate_polygon

class StaticManager(object):

    def __init__(self, idx: Any, section: STSection):
        self.id = idx
        self.section = section
        self.category = 'static'

        agents: List[STAgent] = self.section.agents
        agents_states = list()
        for agent in agents:
            agents_states.append(AgentState(
                agent.id,
                self.category,
                agent.route[0].x,
                agent.route[0].y,
                agent.route[0].heading,
                0.0,
                0.0,
                0.0,
                length=agent.agent_type.length,
                width=agent.agent_type.width,
                height=agent.agent_type.height,
                back2center=agent.agent_type.back2center
            ))
        self.agents_states = agents_states

    def _get_perception_obstacle(self, state: AgentState) -> Tuple[PerceptionObstacle, Polygon]:
        loc = PointENU(x=state.x, y=state.y)
        position = Point3D(x=loc.x, y=loc.y, z=loc.z)
        heading = state.heading

        obj_length = state.length
        obj_width = state.width
        obj_height = state.height

        obs_polygon, polygon_point = generate_polygon(position.x,
                                                      position.y,
                                                      heading,
                                                      front_l=obj_height / 2.0,
                                                      back_l=obj_height / 2.0,
                                                      width=obj_width,
                                                      z=position.z)

        obs = PerceptionObstacle(
            id=state.id,
            position=position,
            theta=heading,
            velocity=Point3D(x=0, y=0, z=0),
            acceleration=Point3D(x=0, y=0, z=0),
            length=obj_length,
            width=obj_width,
            height=obj_height,
            type=PerceptionObstacle.UNKNOWN_UNMOVABLE,
            timestamp=time.time(),
            tracking_time=1.0,
            polygon_point=polygon_point
        )
        return obs, obs_polygon

    def run_step(self, curr_time: float) -> Tuple[List[PerceptionObstacle], List[Polygon], List[AgentState]]:
        apollo_perception = list()
        polygons = list()
        for index, agent_state in enumerate(self.agents_states):
            obs_perception, obs_polygon = self._get_perception_obstacle(agent_state)
            apollo_perception.append(obs_perception)
            polygons.append(obs_polygon)
        return apollo_perception, polygons, self.agents_states