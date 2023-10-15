import math
import random

import carla
import numpy as np
import pygame
import shapely


class NetworkDevice:
    """Network device attached to a Carla Actor. This device enables communication with other devices, which sends its
       own trajectory and receives trajectories from other actors in the environment. The sending range is limited.
   """
    def __init__(self, actor: carla.Actor, max_range: int, id_num: int, max_deceleration: int, scaling_factor: float,
                 max_time_diff: float):
        """
        :param actor: Carla actor which receives this network device.
        :param max_range: Maximum receiving range. Does not influence sending range!
        :param id_num: Unique device ID number.
        :param max_deceleration: Maximum deceleration power of the actor.
        :param scaling_factor: Scaling factor solely for visualization purposes.
        :param max_time_diff: Maximum acceptable differing time of arrival of two actors before the situation becomes
                              critical.
        """
        self.actor = actor
        self.max_range = max_range
        self.id = id_num
        self.prev_dist = None           # Previous determined distance to the critical point.
        self.deceleration = 0.0
        self.intersection_point = None  # The critical point where two trajectories intersect.
        self.critical = False           # Situation is critical.
        self.max_deceleration = max_deceleration
        self.scaling_factor = scaling_factor
        self.max_time_diff = max_time_diff

    def force_stopping(self):
        """Forces the actor to stop to avoid the crash at the critical point."""
        if self.actor.type_id.startswith("walker"):
            self.deceleration = 1.0     # Stops the pedestrian immediately.
        elif self.actor.type_id.startswith("vehicle"):
            self.deceleration = ((self.actor.get_velocity().length() ** 2)
                                 / (2 * self.prev_dist * self.max_deceleration))
        else:
            print("No matching type found for '{}'".format(self.actor.type_id))

    def receive(self, other_actor):
        """Receives information of another actor.
        :param other_actor: Carla actor.
        """
        # Extend the trajectory vector by the scaling factor.
        start = (self.actor.get_transform().location.x, self.actor.get_transform().location.y)
        end = self.actor.get_transform().get_forward_vector()
        end = (int(start[0] + end.x * self.scaling_factor), int(start[1] + end.y * self.scaling_factor))
        this_line = shapely.LineString([start, end])    # Trajectory vector of this actor.

        start = (other_actor.get_transform().location.x, other_actor.get_transform().location.y)
        end = other_actor.get_transform().get_forward_vector()
        end = (int(start[0] + end.x * self.scaling_factor), int(start[1] + end.y * self.scaling_factor))
        other_line = shapely.LineString([start, end])   # Trajectory vector of the other vector.

        if this_line.intersects(other_line):
            # Situation can't be critical, if one of the actors isn't moving.
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

            # Situation is critical, if the difference between both time of arrivals is below the threshold. Other
            # checks are necessary to avoid false positives.
            time_difference = abs(this_time_of_arrival - other_time_of_arrival)
            if (self.prev_dist is not None and self.prev_dist > this_distance
                    and time_difference < self.max_time_diff):
                self.critical = True
            self.prev_dist = this_distance
        else:
            self.critical = False
            self.intersection_point = None


class NetworkEnvironment:
    """Manages all network devices and is responsible that other devices receive information from the actors."""

    def __init__(self, display_man, display_pos, vis_line_width: int, vis_trajectory_window_x: int,
                 vis_trajectory_window_y: int, vis_point_radius: int, vis_scaling_factor_x: int,
                 vis_scaling_factor_y: int, vis_sending_range: int):
        """
        :param display_man: Display manager instance.
        :param display_pos: Grid position inside the display manager.
        :param vis_line_width: Line width of the trajectory for the trajectory visualizer.
        :param vis_trajectory_window_x: Window width of the trajectory visualizer.
        :param vis_trajectory_window_y: Window height of the trajectory visualizer.
        :param vis_point_radius: Point radius of the actor positions and trajectory point.
        :param vis_scaling_factor_x: Stretches x-directions by this factor for the trajectory visualizer.
        :param vis_scaling_factor_y: Stretches y-directions by this factor for the trajectory visualizer.
        :param vis_sending_range: Line radius for the sending range inside the trajectory visualizer.
        """
        self.devices = []
        self.surface = None     # The surface which we draw things onto.
        self.display_man = display_man
        self.display_man.add_window(self)
        self.display_pos = display_pos
        self.stopping_actor_id = None
        self.vis_line_width = vis_line_width
        self.vis_trajectory_window_x = vis_trajectory_window_x
        self.vis_trajectory_window_y = vis_trajectory_window_y
        self.vis_point_radius = vis_point_radius
        self.vis_scaling_factor_x = vis_scaling_factor_x
        self.vis_scaling_factor_y = vis_scaling_factor_y
        self.vis_sending_range = vis_sending_range

    def add_device(self, device: NetworkDevice):
        """Registers a network device.
        :param device: Network device instance.
        """
        self.devices.append(device)

    def check_broadcasts(self):
        """Iteration step: check if devices in range can receive a message from other devices."""
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
        """Display trajectories of all actors relative to the ego vehicle (the point of interest).
        :param ego_vehicle: Carla ego vehicle actor.
        """
        if self.display_man.render_enabled():
            # Initialize drawing surface.
            self.surface = pygame.surface.Surface((self.vis_trajectory_window_x, self.vis_trajectory_window_y))
            self.surface.fill((255, 255, 255))

            ego_pos = ego_vehicle.get_location()
            ego_pos = (ego_pos.x, ego_pos.y)
            detected_hazards = []
            for dev in self.devices:
                if dev.critical:
                    detected_hazards.append(dev.id)

                # Get trajectories of both actors and determine visualization color.
                start = dev.actor.get_transform().location
                if dev.actor.type_id.startswith("walker"):
                    color = "blue"
                    start = (int((start.x - ego_pos[0]) * self.vis_scaling_factor_x
                                 + self.vis_trajectory_window_x // 2),
                             int((start.y - ego_pos[1]) * self.vis_scaling_factor_y
                                 + self.vis_trajectory_window_y // 2))
                    end = dev.actor.get_transform().get_forward_vector() * 200.0    # Just a scaling factor.
                else:
                    color = "black"
                    start = (int(start.x - ego_pos[0] + self.vis_trajectory_window_x // 2),
                             int(start.y - ego_pos[1] + self.vis_trajectory_window_y // 2))
                    end = dev.actor.get_transform().get_forward_vector() * (self.vis_trajectory_window_x / 2)
                end = (int(start[0] + end.x), int(start[1] + end.y))

                if self.stopping_actor_id is not None:
                    if dev.id == self.stopping_actor_id:
                        color = "red"
                    else:
                        color = "green"

                # Display trajectories as dashed lines.
                dl = 10     # Dashed line size.
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
                    pygame.draw.line(self.surface, color, p1, p2, self.vis_line_width)

                # Draw position of actor and maximum sending range.
                pygame.draw.circle(self.surface, color, start, self.vis_point_radius)
                pygame.draw.circle(self.surface, color, start, radius=dev.max_range * self.vis_scaling_factor_x,
                                   width=self.vis_sending_range)
                if dev.intersection_point is not None:
                    # Draw intersection point. Green: OK, red: critical.
                    color = "green" if not dev.critical else "red"
                    start = (
                        int((dev.intersection_point[0] - ego_pos[0]) * self.vis_scaling_factor_x
                            + self.vis_trajectory_window_x // 2),
                        int((dev.intersection_point[1] - ego_pos[1]) * self.vis_scaling_factor_y
                            + self.vis_trajectory_window_y // 2))
                    pygame.draw.circle(self.surface, color, start, self.vis_point_radius)

            pygame.display.flip()
            if len(detected_hazards) == 0:
                return
            # If multiple actors detect a hazard, decide which actor stops. The other one keeps moving.
            if self.stopping_actor_id is None:
                self.stopping_actor_id = random.choice(detected_hazards)
            self.devices[self.stopping_actor_id - 1].force_stopping()

    def render(self):
        """Renders a new frame in every step to visualize the onboard camera image and trajectories."""
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        """Action performed if the code stops."""
        pass
