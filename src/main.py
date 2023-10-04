#!/usr/bin/env python

import argparse
import math
import os
import sys
import yaml

import carla
import numpy as np
import pygame
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
import shapely

from yolov5.yolo_interface import detect, load_model


class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        # TODO
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (1120, 0)
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0] / self.grid_size[1]), int(self.window_size[1] / self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display is not None


class NetworkDevice:
    def __init__(self, actor: carla.Actor, max_range: int, id_num: int):
        self.actor = actor
        self.max_range = max_range
        self.id = id_num
        self.prev_dist = None
        self.deceleration = 0.0

    def receive(self, trajectory):
        start = (self.actor.get_transform().location.x, self.actor.get_transform().location.y)
        end = self.actor.get_transform().get_forward_vector()   # TODO: scaling factor
        end = (int(start[0] + end.x * 100.0), int(start[1] + end.y * 100.0))
        line1 = shapely.LineString([start, end])

        start = (trajectory.location.x, trajectory.location.y)
        end = trajectory.get_forward_vector()  # TODO: scaling factor
        end = (int(start[0] + end.x * 100.0), int(start[1] + end.y * 100.0))
        line2 = shapely.LineString([start, end])


        # TODO: determine whether there will be a crash or not. Not just based on hardcoded scenarios.
        max_deceleration = 200     # TODO: no idea how to get the right value.
        if line1.intersects(line2):
            intersection = line1.intersection(line2)
            actor_point = shapely.Point(self.actor.get_transform().location.x, self.actor.get_transform().location.y)
            distance = intersection.distance(actor_point)
            if self.prev_dist is not None and self.prev_dist > distance:
                self.deceleration = (self.actor.get_velocity().length() ** 2) / (2 * distance * max_deceleration)
            self.prev_dist = distance


class NetworkEnvironment:

    def __init__(self, display_man, display_pos):
        self.devices = []
        self.surface = None
        self.display_man = display_man
        self.display_man.add_sensor(self)
        self.display_pos = display_pos

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
                    self.devices[i].receive(self.devices[j].actor.get_transform())
                if dist <= self.devices[j].max_range:
                    self.devices[j].receive(self.devices[i].actor.get_transform())
                j += 1

    def display_trajectories(self, ego_vehicle):
        # TODO: Many hardcoded values.
        screen_size_x = 800
        screen_size_y = 480
        radius = 6
        visual_scaling_factor_x = 4
        visual_scaling_factor_y = 20
        if self.display_man.render_enabled():
            self.surface = pygame.surface.Surface((screen_size_x, screen_size_y))
            self.surface.fill((255, 255, 255))
            ego_pos = ego_vehicle.get_location()
            ego_pos = (ego_pos.x, ego_pos.y)
            for dev in self.devices:
                start = dev.actor.get_transform().location
                if dev.actor.type_id.startswith("walker"):
                    color = "red"
                    start = (int((start.x - ego_pos[0]) * visual_scaling_factor_x + screen_size_x // 2),
                             int((start.y - ego_pos[1]) * visual_scaling_factor_y + screen_size_y // 2))
                    end = dev.actor.get_transform().get_forward_vector() * 200.0
                else:
                    color = "blue"
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
                pygame.draw.circle(self.surface, color, start, radius=dev.max_range*visual_scaling_factor_x, width=1)
            pygame.display.flip()

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        pass


class SensorManager:
    def __init__(self, world, display_man, transform, attached, sensor_options, display_pos):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(transform, attached, sensor_options)
        self.display_man.add_sensor(self)
        self.ready = False
        self.model, self.stride, self.names = load_model(parameters["yolo"]["device"], parameters["yolo"]["data"],
                                                         parameters["yolo"]["weights"], parameters["yolo"]["half"],
                                                         parameters["yolo"]["img_size"])
        self.ready = True
        self.people_list = []

    def init_sensor(self, transform, attached, sensor_options):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # TODO
        camera_bp.set_attribute('image_size_x', str(800))
        camera_bp.set_attribute('image_size_y', str(640))

        for key in sensor_options:
            camera_bp.set_attribute(key, sensor_options[key])

        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
        camera.listen(self.save_rgb_image)

        return camera

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        annotated, self.people_list = detect(self.model, array, parameters["yolo"]["img_size"],
                                             parameters["yolo"]["conf_thres"], parameters["yolo"]["iou_thres"],
                                             parameters["yolo"]["max_det"], self.names)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(annotated.swapaxes(0, 1))

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


def run_simulation(client, sync=True):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    display_manager = None
    actor_list = []
    world = client.get_world()
    original_settings = world.get_settings()

    try:
        # Getting the world and
        if sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            # settings.fixed_delta_seconds = 0.05    # Everything is too fast
            world.apply_settings(settings)

        # Set spectator viewpoint.
        spectator: carla.Actor = world.get_spectator()
        location = carla.Location(x=-14.856689, y=131.858612, z=7.222219)
        rotation = carla.Rotation(pitch=-31.648739, yaw=4.472778, roll=0.000018)
        transform = carla.Transform(location, rotation)
        spectator.set_transform(transform)

        # Spawn ego vehicle.
        vehicle_blueprint = world.get_blueprint_library().find(parameters["carla"]["vehicle_model"])
        ego_location = carla.Location(x=65, y=130.5, z=0.600000)
        ego_rotation = carla.Rotation(pitch=0.0, yaw=-179.494202, roll=0.0)
        ego_position = carla.Transform(ego_location, ego_rotation)
        ego_vehicle: carla.Vehicle = world.spawn_actor(vehicle_blueprint, ego_position)
        actor_list.append(ego_vehicle)

        pedestrian_blueprint = None
        for ped in world.get_blueprint_library().filter("walker"):
            if ped.id == parameters["carla"]["pedestrian_model"]:
                pedestrian_blueprint = ped
                break
        if pedestrian_blueprint is None:
            print("Couldn't find pedestrian blueprint. It should be a child model.")
            sys.exit(1)
        pedestrian_location = carla.Location(x=-10.0, y=125.0, z=0.6)
        pedestrian_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        pedestrian_position = carla.Transform(pedestrian_location, pedestrian_rotation)
        pedestrian: carla.Actor = world.spawn_actor(pedestrian_blueprint, pedestrian_position)
        actor_list.append(pedestrian)

        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        if args.connected_mobility:
            # TODO.
            display_manager = DisplayManager(grid_size=[2, 1], window_size=[parameters["carla"]["screen_width"],
                                                                            parameters["carla"]["screen_height"] + 480])
        else:
            display_manager = DisplayManager(grid_size=[1, 1], window_size=[parameters["carla"]["screen_width"],
                                                                            parameters["carla"]["screen_height"]])

        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position,
        cam = SensorManager(world, display_manager,
                            carla.Transform(carla.Location(x=0.5, z=1.5)),
                            ego_vehicle, {}, display_pos=[0, 0])
        while not cam.ready:
            pass

        vehicle_control: carla.VehicleControl = carla.VehicleControl()
        vehicle_control.manual_gear_shift = False

        pedestrian_control: carla.WalkerControl = carla.WalkerControl()
        pedestrian_control.speed = 1.5
        pedestrian_control.direction.x = 1
        pedestrian_control.direction.z = 0
        pedestrian.apply_control(pedestrian_control)

        # Simulation loop
        call_exit = False
        braking = False
        dist_printed = False

        if args.connected_mobility:
            env = NetworkEnvironment(display_manager, [1, 0])

            # TODO: make configurable
            ped_device = NetworkDevice(pedestrian, 20, 1)
            vehicle_device = NetworkDevice(ego_vehicle, 55, 2)
            env.add_device(ped_device)
            env.add_device(vehicle_device)
        while True:
            # Carla Tick
            if sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            if args.connected_mobility:
                env.display_trajectories(ego_vehicle)
                env.check_broadcasts()

            velocity = ego_vehicle.get_velocity().length() * 3.6
            if velocity < 50.0 and not braking:
                vehicle_control.throttle = 0.7
            else:
                vehicle_control.throttle = 0.0

            if not args.connected_mobility:
                for person in cam.people_list:
                    # TODO: determine whether person is on the road (and relevant).
                    on_the_road = True
                    if on_the_road and not braking:
                        # TODO: ofc this is crap.
                        ego_loc = ego_vehicle.get_location()
                        ped_loc = pedestrian.get_location()
                        dist = math.sqrt((ped_loc.x - ego_loc.x) ** 2 + (ped_loc.y - ego_loc.y) ** 2)
                        if dist < 30.0:
                            braking = True
                            vehicle_control.brake = 1
                            vehicle_control.throttle = 0

            if args.connected_mobility:
                if vehicle_device.deceleration > 0.005:
                    # TODO: Release after danger is over.
                    braking = True
                    print(vehicle_device.deceleration)
                    vehicle_control.brake = vehicle_device.deceleration
                    vehicle_control.throttle = 0
            print(vehicle_control)
            ego_vehicle.apply_control(vehicle_control)

            if velocity < 0.1 and braking and not dist_printed:
                dist_printed = True
                ego_loc = ego_vehicle.get_location()
                ped_loc = pedestrian.get_location()
                dist = math.sqrt((ped_loc.x - ego_loc.x) ** 2 + (ped_loc.y - ego_loc.y) ** 2)
                print("Vehicle-Pedestrian Distance: {}".format(dist))

            if dist_printed:
                pedestrian_control.direction.x = 0
                pedestrian_control.direction.y = 0
                pedestrian.apply_control(pedestrian_control)
            elif pedestrian.get_location().x >= -3.8:
                pedestrian_control.direction.x = 0
                pedestrian_control.direction.y = 1
                pedestrian.apply_control(pedestrian_control)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                break

    finally:
        if display_manager:
            display_manager.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        world.apply_settings(original_settings)


def main():
    try:
        client = carla.Client(parameters["carla"]["host"], parameters["carla"]["port"])
        client.set_timeout(parameters["carla"]["timeout"])
        run_simulation(client)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    with open("config.yaml", "r") as cfg_file:
        argparser = argparse.ArgumentParser(description="CARLA Autonomous Driving vs Connected Mobility")
        argparser.add_argument("--connected_mobility", action="store_true", help="Enable connected mobility")
        args = argparser.parse_args()
        parameters = yaml.safe_load(cfg_file)
        main()
