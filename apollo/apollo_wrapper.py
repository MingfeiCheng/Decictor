import time

from omegaconf import DictConfig
from typing import List, Optional, Set, Tuple, TypeVar

from apollo.docker_container import ApolloContainer
from apollo.cyber_bridge import Topics
from apollo.map_parser import MapParser

from apollo_modules.modules.common.proto.geometry_pb2 import Point3D
from apollo_modules.modules.common.proto.header_pb2 import Header
from apollo_modules.modules.localization.proto.localization_pb2 import LocalizationEstimate
from apollo_modules.modules.localization.proto.pose_pb2 import Pose
from apollo_modules.modules.planning.proto.planning_pb2 import ADCTrajectory
from apollo_modules.modules.routing.proto.routing_pb2 import LaneWaypoint, RoutingRequest

from scenario.config.common import Waypoint

WaypointClass = TypeVar("WaypointClass", bound=Waypoint)

from loguru import logger

class ApolloWrapper:
    """
    Class to manage and run an Apollo instance
    """
    # config
    nid: int # unique apollo id
    container: ApolloContainer # apollo container controls the runner instance
    route: List # List of route 0-start 1-dest
    time_limit: float  # time max running
    start_time: float # time to start

    # some flags
    routing_response: bool

    # callback data
    localization: Optional[LocalizationEstimate]
    planning: Optional[ADCTrajectory]

    # inner records
    __min_distance: Optional[float] # collision
    __decisions: Set[Tuple] # decision
    __coords: List[Tuple] # real localization
    __dist2dest: Optional[float] #

    def __init__(self,
                 nid: int,
                 cfg: DictConfig,
                 ctn: ApolloContainer,
                 route: List[WaypointClass]
                 ) -> None:
        """
        Wrapper for apollo (apollo container)
        This is used for controlling apollo behaviors/decisions:
            (1) container/bridge
            (2) route
            (3) oracles: collision & timeout
        """
        self.nid = nid
        self.container = ctn
        self.route = route

        self.time_limit = cfg.time_limit
        self.stuck_time_limit = cfg.stuck_time_limit
        self.start_time = cfg.start_time

        self.routing_response = False

        self.planning = None
        self.localization = None

        self.__min_distance = None
        self.__coords = list()
        self.__dist2dest = None
        self.__violations = list()

    def initialize(self):
        """
        Resets and initializes all necessary modules of Apollo
        """
        self.register_publishers()
        self.register_subscribers()

        self.send_initial_localization()
        self.container.start_sim_control()

        # initialize class variables
        self.routing_response = False
        self.planning = None
        self.localization = None

        self.__min_distance = None
        self.__coords = list()

        # make sure all connection before this step
        self.container.bridge.spin()
        logger.info('Initialized Apollo Wrapper')

    def register_publishers(self):
        """
        Register publishers for the cyberRT communication
        """
        for c in [Topics.Localization, Topics.Obstacles, Topics.TrafficLight, Topics.RoutingRequest]:
            self.container.bridge.add_publisher(c)

    def register_subscribers(self):
        """
        Register subscribers for the cyberRT communication
        """
        def localization_cb(data):
            """
            Callback function when localization message is received
            """
            self.localization = data
            self.__coords.append((data.pose.position.x, data.pose.position.y))

        self.container.bridge.add_subscriber(Topics.Localization, localization_cb)

    def send_initial_localization(self):
        """
        Send the instance's initial location to cyberRT
        """
        ma = MapParser.get_instance()

        coord, heading = ma.get_coordinate_and_heading(self.route[0].lane_id, self.route[0].s)
        loc = LocalizationEstimate(
            header=Header(
                timestamp_sec=time.time(),
                module_name="MAGGIE",
                sequence_num=0
            ),
            pose=Pose(
                position=coord,
                heading=heading,
                linear_velocity=Point3D(x=0, y=0, z=0)
            )
        )

        for i in range(400):
            loc.header.sequence_num = i
            self.container.bridge.publish(Topics.Localization, loc.SerializeToString())
            # time.sleep(0.1)

    def send_route(self, t: float):
        """
        Send the instance's routing request to cyberRT
        """
        if t < self.start_time or self.routing_response:
            return

        ma = MapParser.get_instance()
        coord, heading = ma.get_coordinate_and_heading(self.route[0].lane_id, self.route[0].s)

        rr = RoutingRequest(
            header=Header(
                timestamp_sec=time.time(),
                module_name="MAGGIE",
                sequence_num=0
            ),
            waypoint=[
                LaneWaypoint(
                    pose=coord,
                    heading=heading
                )
            ] + [
                LaneWaypoint(
                    id=x.lane_id,
                    s=x.s,
                ) for x in self.route
            ]
        )

        self.container.bridge.publish(
            Topics.RoutingRequest, rr.SerializeToString()
        )

        self.routing_response = True

    def set_min_distance(self, d: float):
        """
        Updates the minimum distance between this distance and another object if the
        argument passed in is smaller than the current min distance

        :param float d: the distance between this instance and another object.
        """
        if self.__min_distance is None:
            self.__min_distance = d
        elif d < self.__min_distance:
            self.__min_distance = d
            # logger.debug(f'set min_distance: {self.__min_distance}')

    def set_dist2dest(self, d: float):
        self.__dist2dest = d

    def set_violations(self, v: str):
        self.__violations.append(v)
        self.__violations = list(set(self.__violations))

    def get_destination(self) -> WaypointClass:
        return self.route[-1]

    def get_min_distance(self) -> float:
        """
        Get the minimum distance this instance ever reached w.r.t. another
        object. e.g., 0 if a collision occurred error here

        :returns: the minimum distance between this Apollo instance and another object
        :rtype: float
        """
        if self.__min_distance is None:
            return 10000
        return self.__min_distance

    def get_dist2dest(self) -> float:
        if not self.__dist2dest:
            return 10000
        return self.__dist2dest

    def get_trajectory(self) -> List[Tuple]:
        """
        Get the points traversed by this Apollo instance

        :returns: list of coordinates traversed by this Apollo instance
        :rtype: List[Tuple[float, float]]
        """
        return self.__coords

    def get_violations(self) -> List:
        return self.__violations