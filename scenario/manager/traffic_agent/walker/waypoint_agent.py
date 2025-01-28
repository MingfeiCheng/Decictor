import math

import numpy as np

from typing import Any, Tuple, Optional
from collections import deque

from apollo_modules.modules.common.proto.geometry_pb2 import PointENU

from scenario.config.common import MotionWaypoint
from scenario.config.section.walker import WAAgent

# TODO: add repeat
class WaypointAgent(object):

    def __init__(self, agent: WAAgent):
        """
        Set up actor and local planner
        NOTE: input speed is m/s
        """
        self._agent = agent

        self._waypoints_queue = deque(agent.route)

        self._agent_height = agent.agent_type.height
        self._agent_width = agent.agent_type.width

        self._curr_x = self._agent.route[0].x
        self._curr_y = self._agent.route[0].x
        self._curr_heading = self._agent.route[0].heading
        self._curr_speed = 0.0

        self._base_min_distance = 1.0

    def _get_next_waypoint(self) -> Optional[MotionWaypoint]:
        # @Mingfei finish this

        # Purge the queue of obsolete waypoints
        min_distance = self._base_min_distance + 0.5 * self._curr_speed

        num_waypoint_removed = 0
        for waypoint in self._waypoints_queue:

            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by

            if np.sqrt((waypoint.x - self._curr_x) ** 2 + (waypoint.y - self._curr_y) ** 2) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        if len(self._waypoints_queue) == 0:
            return None

        return self._waypoints_queue[0]

    def run_step(self,
                 delta_time: float,
                 agent_id: Any,
                 running=True) -> Tuple[PointENU, float, float]:
        _ = agent_id # TODO: add collision? -> No
        if not running:
            position = PointENU(x=self._curr_x, y=self._curr_y)
            heading = self._curr_heading
            speed = 0.0
            return position, heading, speed

        # step 1: get next waypoint
        target_wp = self._get_next_waypoint()
        if target_wp is None:
            # do not contain any other next
            # @mingfei todo: improve this -> continue or something else
            # current stop when finished
            position = PointENU(x=self._curr_x, y=self._curr_y)
            heading = self._curr_heading
            speed = 0.0  # self._curr_state.speed # @mingfei please check later
            return position, heading, speed

        # todo: add traffic rules
        # NOTE: only consider normal speed
        target_speed = target_wp.speed
        if target_speed > self._curr_speed:
            acc = 0.5
            new_speed = self._curr_speed + delta_time * acc
            new_speed = np.clip(new_speed, 0.0, target_speed)
        elif target_speed < self._curr_speed:
            acc = -0.5
            new_speed = self._curr_speed + delta_time * acc
            new_speed = np.clip(new_speed, target_speed, None)
        else:
            new_speed = target_speed

        new_x = self._curr_x + self._curr_speed * math.cos(self._curr_heading) * delta_time
        new_y = self._curr_y + self._curr_speed * math.sin(self._curr_heading) * delta_time

        target_x = target_wp.x
        target_y = target_wp.y

        new_heading = math.atan2(target_y - self._curr_y, target_x - self._curr_x)

        self._curr_x = new_x
        self._curr_y = new_y
        self._curr_heading = new_heading
        self._curr_speed = new_speed

        position = PointENU(x=self._curr_x, y=self._curr_y)

        return position, self._curr_heading, self._curr_speed

