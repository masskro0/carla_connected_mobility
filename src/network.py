import math
import random

import carla
import numpy as np
import pygame
import shapely


class NetworkDevice:
    def __init__(self, actor: carla.Actor, max_range: int, id_num: int, max_deceleration: int, scaling_factor: float,
                 max_time_diff: float):
        self.actor = actor
        self.max_range = max_range
        self.id = id_num
        self.prev_dist = None
        self.deceleration = 0.0
        self.intersection_point = None
        self.critical = False
        self.max_deceleration = max_deceleration
        self.scaling_factor = scaling_factor
        self.max_time_diff = max_time_diff

    def force_stopping(self):
        if self.actor.type_id.startswith("walker"):
            self.deceleration = 1.0
        elif self.actor.type_id.startswith("vehicle"):
            self.deceleration = ((self.actor.get_velocity().length() ** 2)
                                 / (2 * self.prev_dist * self.max_deceleration))
        else:
            print("No matching type found for '{}'".format(self.actor.type_id))

    def receive(self, other_actor):
        start = (self.actor.get_transform().location.x, self.actor.get_transform().location.y)
        end = self.actor.get_transform().get_forward_vector()
        end = (int(start[0] + end.x * self.scaling_factor), int(start[1] + end.y * self.scaling_factor))
        this_line = shapely.LineString([start, end])

        start = (other_actor.get_transform().location.x, other_actor.get_transform().location.y)
        end = other_actor.get_transform().get_forward_vector()
        end = (int(start[0] + end.x * self.scaling_factor), int(start[1] + end.y * self.scaling_factor))
        other_line = shapely.LineString([start, end])

        if this_line.intersects(other_line):
            if self.actor.get_velocity().length() == 0 or other_actor.get_velocity().length() == 0:
                return
            intersection = this_line.intersection(other_line)
            self.intersection_point = (intersection.x, intersection.y)
            actor_point = shapely.Point(self.actor.get_transform().location.x, self.actor.get_transform().location.y)
            this_distance = intersection.distance(actor_point)
            this_time_of_arrival = this_distance / (self.actor.get_velocity().length())

            other_point = shapely.Point(
                (other_actor.get_transform().location.x, other_actor.get_transform().location.y))
            other_distance = intersection.distance(other_point)
            other_time_of_arrival = other_distance / (other_actor.get_velocity().length())

            time_difference = abs(this_time_of_arrival - other_time_of_arrival)
            if (self.prev_dist is not None and self.prev_dist > this_distance
                    and time_difference < self.max_time_diff):
                self.critical = True
            self.prev_dist = this_distance
        else:
            self.critical = False
            self.intersection_point = None


class NetworkEnvironment:

    def __init__(self, display_man, display_pos, vis_trajectory_window_x: int, vis_trajectory_window_y: int,
                 vis_point_radius: int):
        self.devices = []
        self.surface = None
        self.display_man = display_man
        self.display_man.add_sensor(self)
        self.display_pos = display_pos
        self.stopping_actor_id = None
        self.vis_trajectory_window_x = vis_trajectory_window_x
        self.vis_trajectory_window_y = vis_trajectory_window_y
        self.vis_point_radius = vis_point_radius

    def add_device(self, device: NetworkDevice):
        self.devices.append(device)

    def check_broadcasts(self):
        for i in range(len(self.devices)):
            j = i + 1
            while j < len(self.devices):
                dev1_loc = self.devices[i].actor.get_transform().location
                dev2_loc = self.devices[j].actor.get_transform().location
                dist = math.sqrt((dev1_loc.x - dev2_loc.x) ** 2 + (dev1_loc.y - dev2_loc.y) ** 2)
                if dist <= self.devices[i].max_range:
                    self.devices[i].receive(self.devices[j].actor)
                if dist <= self.devices[j].max_range:
                    self.devices[j].receive(self.devices[i].actor)
                j += 1

    def display_trajectories(self, ego_vehicle):
        # TODO: Many hardcoded values.
        screen_size_x = self.vis_trajectory_window_x
        screen_size_y = self.vis_trajectory_window_y
        radius = self.vis_point_radius
        visual_scaling_factor_x = 4
        visual_scaling_factor_y = 20
        if self.display_man.render_enabled():
            self.surface = pygame.surface.Surface((screen_size_x, screen_size_y))
            self.surface.fill((255, 255, 255))
            ego_pos = ego_vehicle.get_location()
            ego_pos = (ego_pos.x, ego_pos.y)
            detected_hazards = []
            for dev in self.devices:
                if dev.critical:
                    detected_hazards.append(dev.id)
                start = dev.actor.get_transform().location
                if dev.actor.type_id.startswith("walker"):
                    color = "blue"
                    start = (int((start.x - ego_pos[0]) * visual_scaling_factor_x + screen_size_x // 2),
                             int((start.y - ego_pos[1]) * visual_scaling_factor_y + screen_size_y // 2))
                    end = dev.actor.get_transform().get_forward_vector() * 200.0
                else:
                    color = "black"
                    start = (int(start.x - ego_pos[0] + screen_size_x // 2),
                             int(start.y - ego_pos[1] + screen_size_y // 2))
                    end = dev.actor.get_transform().get_forward_vector() * (screen_size_x / 2)
                end = (int(start[0] + end.x), int(start[1] + end.y))

                dl = 10
                if start[0] == end[0]:
                    ycoords = [y for y in range(start[1], end[1], dl if start[1] < end[1] else -dl)]
                    xcoords = [start[0]] * len(ycoords)
                elif start[1] == end[1]:
                    xcoords = [x for x in range(start[0], end[0], dl if start[0] < end[0] else -dl)]
                    ycoords = [start[1]] * len(xcoords)
                else:
                    a = abs(end[0] - start[0])
                    b = abs(end[1] - start[1])
                    c = round(math.sqrt(a ** 2 + b ** 2))
                    dx = dl * a / c
                    dy = dl * b / c

                    xcoords = [x for x in np.arange(start[0], end[0], dx if start[0] < end[0] else -dx)]
                    ycoords = [y for y in np.arange(start[1], end[1], dy if start[1] < end[1] else -dy)]

                next_coords = list(zip(xcoords[1::2], ycoords[1::2]))
                last_coords = list(zip(xcoords[0::2], ycoords[0::2]))
                for (x1, y1), (x2, y2) in zip(next_coords, last_coords):
                    p1 = (round(x1), round(y1))
                    p2 = (round(x2), round(y2))
                    pygame.draw.line(self.surface, color, p1, p2, 3)
                pygame.draw.circle(self.surface, color, start, radius)
                pygame.draw.circle(self.surface, color, start, radius=dev.max_range * visual_scaling_factor_x, width=1)
                if dev.intersection_point is not None and dev.id == 2:
                    # TODO: Hardcoded ID.
                    color = "green" if not dev.critical else "red"
                    start = (
                        int((dev.intersection_point[0] - ego_pos[0]) * visual_scaling_factor_x + screen_size_x // 2),
                        int((dev.intersection_point[1] - ego_pos[1]) * visual_scaling_factor_y + screen_size_y // 2))
                    pygame.draw.circle(self.surface, color, start, radius)

            pygame.display.flip()
            if len(detected_hazards) == 0:
                return
            if self.stopping_actor_id is None:
                self.stopping_actor_id = random.choice(detected_hazards)
            self.devices[self.stopping_actor_id - 1].force_stopping()

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        pass
