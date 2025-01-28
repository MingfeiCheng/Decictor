import math
import numpy as np

from collections import deque
from shapely.geometry import Polygon
from typing import Any, List
from dataclasses import dataclass

def right_rotation(coord, theta):
    """
    theta : degree
    """
    # theta = math.radians(theta)
    x = coord[1]
    y = coord[0]
    x1 = x * math.cos(theta) - y * math.sin(theta)
    y1 = x * math.sin(theta) + y * math.cos(theta)
    return [y1, x1]

@dataclass
class AgentState:
    id: Any
    category: str
    x: float
    y: float
    heading: float
    speed: float
    acceleration: float
    steering: float
    length: float
    width: float
    height: float
    back2center: float
    # travel_s: float # no need

    def get_forward_vector(self) -> List:
        init_vector = [1, 0]
        forward_vector = right_rotation(init_vector, -self.heading)
        return forward_vector

    def get_polygon(self) -> Polygon:
        points = []
        half_w = self.width / 2.0
        if self.category == 'ego':
            assert self.back2center != 0.0
            front_l = self.length - self.back2center
            back_l = -1 * self.back2center
        else:
            front_l = self.length / 2.0
            back_l = -1 * self.length / 2.0
        sin_h = math.sin(self.heading)
        cos_h = math.cos(self.heading)
        vectors = [(front_l * cos_h - half_w * sin_h,
                    front_l * sin_h + half_w * cos_h),
                   (back_l * cos_h - half_w * sin_h,
                    back_l * sin_h + half_w * cos_h),
                   (back_l * cos_h + half_w * sin_h,
                    back_l * sin_h - half_w * cos_h),
                   (front_l * cos_h + half_w * sin_h,
                    front_l * sin_h - half_w * cos_h)]
        for x, y in vectors:
            points.append([self.x + x, self.y + y])
        return Polygon(points)

    def get_front_buffer_polygon(self, buffer: float) -> Polygon:
        points = []
        half_w = self.width / 2.0
        if self.category == 'ego':
            assert self.back2center != 0.0
            front_l = self.length - self.back2center
            back_l = -1 * self.back2center
        else:
            front_l = self.length / 2.0
            back_l = -1 * self.length / 2.0
        front_l += buffer
        sin_h = math.sin(self.heading)
        cos_h = math.cos(self.heading)
        vectors = [(front_l * cos_h - half_w * sin_h,
                    front_l * sin_h + half_w * cos_h),
                   (back_l * cos_h - half_w * sin_h,
                    back_l * sin_h + half_w * cos_h),
                   (back_l * cos_h + half_w * sin_h,
                    back_l * sin_h - half_w * cos_h),
                   (front_l * cos_h + half_w * sin_h,
                    front_l * sin_h - half_w * cos_h)]
        for x, y in vectors:
            points.append([self.x + x, self.y + y])
        # print(f"x: {self.x}, y: {self.y}, heading: {self.heading}, length: {self.length}, width: {self.width}")
        # print(points)
        return Polygon(points)


class KinematicBicycleModel:

    def __init__(self, wheelbase: float):
        self.wheelbase = wheelbase

    def run_step(self, curr_state: AgentState, acceleration: float, steering: float, delta_time: float, max_speed: float = 20.0) -> AgentState:
        if curr_state.speed + delta_time * acceleration >= 0:
            next_speed = curr_state.speed + delta_time * acceleration
        else:
            next_speed = 0.0

        next_speed = np.clip(next_speed, 0.0, max_speed)

        # Compute the angular velocity
        curr_angular_speed = next_speed * math.tan(steering) / self.wheelbase # radius

        # Compute the final state using the discrete time model
        next_x = curr_state.x + curr_state.speed * math.cos(curr_state.heading) * delta_time
        next_y = curr_state.y + curr_state.speed * math.sin(curr_state.heading) * delta_time
        # next_heading = normalise_angle(curr_state.heading + curr_angular_speed * delta_time)
        next_heading = curr_state.heading + curr_angular_speed * delta_time

        if next_heading > math.pi:
            next_heading = next_heading - 2 * math.pi

        if next_heading < -math.pi:
            next_heading = next_heading + 2 * math.pi

        # next_heading = float(np.clip(next_heading, -math.pi, math.pi))
        # return AgentState(new_x, new_y, new_heading, new_speed, state.acceleration, state.steering, new_travel_s)
        return AgentState(curr_state.id,
                          curr_state.category,
                          next_x,
                          next_y,
                          next_heading,
                          next_speed,
                          acceleration,
                          steering,
                          curr_state.length,
                          curr_state.width,
                          curr_state.height,
                          curr_state.back2center)

class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, current_speed: float, target_speed: float, dt: float) -> float:
        return self._pid_control(target_speed, current_speed, dt)

    def _pid_control(self, target_speed: float, current_speed: float, dt: float) -> float:
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt

class PIDLateralController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, current_heading: float, target_heading: float, dt: float) -> float:
        return self._pid_control(target_heading, current_heading, dt)

    def _pid_control(self, target_heading: float, current_heading: float, dt: float) -> float:
        error = target_heading - current_heading

        if error > math.pi:
            error = error - 2 * math.pi
        if error < -math.pi:
            error = error + 2 * math.pi
        error = float(np.clip(error, -1.0, 1.0))
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt