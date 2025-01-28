import math

from typing import Tuple, List
from shapely.geometry import Polygon
from apollo_modules.modules.common.proto.geometry_pb2 import Point3D

def generate_polygon(x: float, y: float, heading: float, front_l: float, back_l: float, width: float, z: float=0.0) -> Tuple[Polygon, List[Point3D]]:
    half_w = width / 2.0
    sin_h = math.sin(heading)
    cos_h = math.cos(heading)
    if back_l > 0:
        back_l = -1 * abs(back_l)
    vectors = [(front_l * cos_h - half_w * sin_h,
                front_l * sin_h + half_w * cos_h),
               (back_l * cos_h - half_w * sin_h,
                back_l * sin_h + half_w * cos_h),
               (back_l * cos_h + half_w * sin_h,
                back_l * sin_h - half_w * cos_h),
               (front_l * cos_h + half_w * sin_h,
                front_l * sin_h - half_w * cos_h)]
    polygon_points = []
    apollo_points = []
    for delta_x, delta_y in vectors:
        polygon_points.append([x + delta_x, y + delta_y])
        p = Point3D()
        p.x = x + delta_x
        p.y = y + delta_y
        p.z = z
        apollo_points.append(p)
    return Polygon(polygon_points), apollo_points