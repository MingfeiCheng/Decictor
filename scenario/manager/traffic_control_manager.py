from time import time
from typing import Any

from apollo_modules.modules.perception.proto.traffic_light_detection_pb2 import TrafficLight, TrafficLightDetection
from scenario.config.section.traffic_light import TLSection

class TrafficControlManager(object):

    def __init__(self, idx: Any, section: TLSection):
        self.id = idx
        self.section = section
        self.last_time = 0.0
        self.category = 'traffic_light'
        self.sequence_num = 0
        self.force_green = False

    def set_force_green(self, force_green: bool):
        self.force_green = force_green

    def get_traffic_lights(self, curr_t: float) -> TrafficLightDetection:
        local_t = curr_t
        while local_t > self.section.duration_g + self.section.duration_y + self.section.duration_r:
            local_t -= (self.section.duration_g + self.section.duration_y + self.section.duration_r)

        # logger.debug(f'Force_green: {self.force_green}')
        if self.force_green:
            config = self.section.get_force_green_config()
        # elif self.section.initial == self.section.final:
        #     config = self.section.initial
        else:
            # rewrite this
            if local_t <= self.section.duration_g:
                # green duration if [0, duration_g]
                config = self.section.initial
            elif self.section.duration_g < local_t <= self.section.duration_g + self.section.duration_y:
                config = self.section.calculate_transition_yellow()
                # yellow duration
                pass
            else:
                #self.section.duration_g + self.section.duration_y < local_t <= self.section.duration_g + self.section.duration_y + self.section.duration_r:
                # buffer duration
                config = self.section.calculate_transition_red()
            # else:
            #     config = self.section.final

        tld = TrafficLightDetection()
        tld.header.timestamp_sec = time()
        tld.header.module_name = "MAGGIE" #"MAGGIE"
        tld.header.sequence_num = self.sequence_num
        self.sequence_num += 1

        for k in config:
            tl = tld.traffic_light.add()
            tl.id = k
            tl.confidence = 1

            if config[k] == 'GREEN':
                tl.color = TrafficLight.GREEN
            elif config[k] == 'YELLOW':
                tl.color = TrafficLight.YELLOW
            else:
                tl.color = TrafficLight.RED

        return tld
