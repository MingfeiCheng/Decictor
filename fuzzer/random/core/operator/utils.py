import math
import random
from typing import List, Optional, Tuple, TypeVar

from shapely.geometry import Point, Polygon, LineString

from apollo.map_parser import MapParser
from scenario.config.lib.agent_type import AgentType

AgentTypeClass = TypeVar("AgentTypeClass", bound=AgentType)

def loc_in_lance(ma: MapParser, pt: Point, lane_pool: List) -> Optional[str]:
    for lane_id in lane_pool:
        lane_polygon = ma.get_lane_polygon(lane_id)
        if lane_polygon.distance(pt) <= 0:
            return lane_id
    return None


def generate_polygon(position: List, shape: List) -> Polygon:
    """
    Generate polygon for a perception obstacle
    position : [x, y, z, heading]
    shape: [length, width, height]
    :returns:
        List with 4 Point3D objects representing the polygon of the obstacle
    :rtype: List[Point3D]
    """
    points = []
    theta = position[3]
    length = shape[0]
    width = shape[1]
    half_l = length / 2.0
    half_w = width / 2.0
    sin_h = math.sin(theta)
    cos_h = math.cos(theta)
    vectors = [(half_l * cos_h - half_w * sin_h,
                half_l * sin_h + half_w * cos_h),
               (-half_l * cos_h - half_w * sin_h,
                - half_l * sin_h + half_w * cos_h),
               (-half_l * cos_h + half_w * sin_h,
                - half_l * sin_h - half_w * cos_h),
               (half_l * cos_h + half_w * sin_h,
                half_l * sin_h - half_w * cos_h)]
    for x, y in vectors:
        p_x = position[0] + x
        p_y = position[1] + y
        points.append([p_x, p_y])

    return Polygon(points)

def get_curve_point_heading(curve: LineString, s: float) -> Tuple[Optional[Point], Optional[float]]:
    ip = curve.interpolate(s)  # a point
    if ip is None or curve is None:
        return None, None

    segments = list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))
    segments.sort(key=lambda x: ip.distance(x))
    line = segments[0]
    x1, x2 = line.xy[0]
    y1, y2 = line.xy[1]

    return ip, math.atan2(y2 - y1, x2 - x1)


def get_segment_point_heading(lane_segment: LineString,
                              global_line: LineString,
                              prohibit_region: Polygon,
                              agent_type: AgentTypeClass,
                              dist2prohibit: float = 0.0,
                              max_tries: int = 50,
                              min_s: Optional[float] = None,
                              max_s: Optional[float] = None) -> Tuple[Optional[Point], Optional[float], Optional[float]]:

    tries = 0
    while tries < max_tries:
        tries += 1
        if min_s is None or max_s is None:
            s = random.uniform(0.1, lane_segment.length - 0.1)
        else:
            s = random.uniform(min_s, max_s)
        local_point = lane_segment.interpolate(s)  # a point
        global_s = global_line.project(local_point)
        point, heading = get_curve_point_heading(global_line, global_s)
        if point is None:
            continue
        tmp_polygon = generate_polygon([point.x, point.y, 0.0, heading],
                                       [agent_type.length, agent_type.width, agent_type.height])

        if tmp_polygon.distance(prohibit_region) <= dist2prohibit:
            continue
        else:
            return point, heading, global_s
    return None, None, None