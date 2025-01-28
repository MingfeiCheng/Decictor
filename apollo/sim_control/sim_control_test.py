import math

import subprocess
from threading import Thread
from time import time, sleep

from cyber.python.cyber_py3 import cyber
from cyber.python.cyber_py3 import cyber_time
from modules.common.proto.geometry_pb2 import Point3D, PointENU
from modules.common.proto.header_pb2 import Header
from modules.routing.proto.routing_pb2 import RoutingRequest, LaneWaypoint
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacle
from modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacles


_s_seq_num = 0


def get_seq_num():
    """
    Return the sequence number
    """
    global _s_seq_num
    _s_seq_num += 1
    return _s_seq_num


def generate_polygon(point, heading, length, width):
    """
    Generate polygon
    """
    points = []
    half_l = length / 2.0
    half_w = width / 2.0
    sin_h = math.sin(heading)
    cos_h = math.cos(heading)
    vectors = [(half_l * cos_h - half_w * sin_h,
                half_l * sin_h + half_w * cos_h),
               (-half_l * cos_h - half_w * sin_h,
                - half_l * sin_h + half_w * cos_h),
               (-half_l * cos_h + half_w * sin_h,
                - half_l * sin_h - half_w * cos_h),
               (half_l * cos_h + half_w * sin_h,
                half_l * sin_h - half_w * cos_h)]
    for x, y in vectors:
        p = Point3D()
        p.x = point.x + x
        p.y = point.y + y
        p.z = point.z
        points.append(p)

    return points


def init_perception():
    """
    Create perception from description
    """
    perception = PerceptionObstacle()
    perception.id = 1
    perception.position.x = 587162.7872082404
    perception.position.y = 4141193.8635936035
    perception.position.z = 0
    perception.theta = 2.880982072755167
    perception.velocity.x = 0
    perception.velocity.y = 0
    perception.velocity.z = 0
    perception.length = 4.70
    perception.width = 2.06
    perception.height = 2.05
    perception.polygon_point.extend(generate_polygon(perception.position,
                                                     perception.theta,
                                                     perception.length,
                                                     perception.width))
    perception.tracking_time = 1.0
    perception.type = PerceptionObstacle.VEHICLE
    perception.timestamp = cyber_time.Time.now().to_sec()

    return perception


def generate_perception():
    """
    Generate perception data
    """
    perceptions = PerceptionObstacles()
    perceptions.header.sequence_num = get_seq_num()
    perceptions.header.module_name = "perception"
    perceptions.header.timestamp_sec = cyber_time.Time.now().to_sec()
    perceptions.perception_obstacle.add().CopyFrom(init_perception())
    return perceptions


def start_dreamview():
    subprocess.run(["/apollo/scripts/bootstrap.sh", "restart"])


def start_modules():
    for module in ['planning', 'prediction', 'routing']:
        subprocess.run(f'/apollo/scripts/{module}.sh start'.split())


def send_localization(localization_writer):
    loc = LocalizationEstimate()
    loc.header.timestamp_sec = time()
    loc.header.module_name = "sim_control_test"
    loc.header.sequence_num = 0
    loc.pose.position.x = 587172.4495365359
    loc.pose.position.y = 4141191.286887972
    loc.pose.heading = 2.880982072755167

    for i in range(5):
        loc.header.sequence_num = i
        localization_writer.write(loc)
        sleep(0.5)


def send_routing_request(routing_writer):
    rr = RoutingRequest(
        header=Header(
            timestamp_sec=time(),
            module_name="sim_control_test",
            sequence_num=0
        ),
        waypoint=[
            LaneWaypoint(
                pose=PointENU(x=587172.4495365359, y=4141191.286887972),
                heading=2.880982072755167
            ),
            LaneWaypoint(
                id='lane_25',
                s=190,
            )
        ]
    )
    routing_writer.write(rr)


def publish_obstacle(perception_writer):
    counter = 0
    while True:
        counter += 0.1
        perception_writer.write(generate_perception())
        sleep(0.1)
        if counter > 20:
            break


if __name__ == '__main__':
    cyber.init()
    node = cyber.Node("sim_control_test")
    pwriter = node.create_writer(
        '/apollo/perception/obstacles', PerceptionObstacles)
    lwriter = node.create_writer(
        '/apollo/localization/pose', LocalizationEstimate
    )
    rwriter = node.create_writer(
        '/apollo/routing_request', RoutingRequest
    )
    # start_dreamview()
    # start_modules()

    while True:
        # start sim control
        subprocess.run(
            "/apollo/modules/sim_control/script.sh start".split()
        )

        send_localization(lwriter)
        t = Thread(target=publish_obstacle, args=(pwriter, ))
        t.start()
        sleep(2)
        send_routing_request(rwriter)
        sleep(20)
        t.join()

        # stop sim control
        subprocess.run(
            "/apollo/modules/sim_control/script.sh stop".split()
        )


# bazel run //modules/sim_control:sim_control_test
