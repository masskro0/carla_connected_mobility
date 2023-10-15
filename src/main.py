#!/usr/bin/env python

import argparse
import math
import sys
import yaml

import carla
import pygame
from pygame.locals import K_ESCAPE
from pygame.locals import K_q

from network import NetworkDevice, NetworkEnvironment
from utils import DisplayManager, YOLOCamera


def run_simulation(client, parameters, sync=True):
    # Display Manager organizes everything we would like to see in a separate window.
    if args.connected_mobility:
        vis_y = parameters["carla"]["vis_trajectory_window_y"]
        display_manager = DisplayManager(grid_size=[2, 1], window_size=[parameters["carla"]["screen_width"],
                                                                        parameters["carla"]["screen_height"] + vis_y])
    else:
        display_manager = DisplayManager(grid_size=[1, 1], window_size=[parameters["carla"]["screen_width"],
                                                                        parameters["carla"]["screen_height"]])

    world = client.get_world()
    original_settings = world.get_settings()
    actor_list = []

    try:
        # Simulation settings.
        if sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            # settings.fixed_delta_seconds = 0.02    # Everything is too fast
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

        # Spawn child.
        pedestrian_blueprint = world.get_blueprint_library().find(parameters["carla"]["pedestrian_model"])
        pedestrian_location = carla.Location(x=-10.0, y=125.0, z=0.6)
        pedestrian_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        pedestrian_position = carla.Transform(pedestrian_location, pedestrian_rotation)
        pedestrian: carla.Actor = world.spawn_actor(pedestrian_blueprint, pedestrian_position)
        actor_list.append(pedestrian)

        # Attach camera to vehicle and load YOLO model.
        cam = YOLOCamera(world, display_manager, carla.Transform(carla.Location(x=0.5, z=1.5)), ego_vehicle, [0, 0],
                         parameters["yolo"]["device"], parameters["yolo"]["data"], parameters["yolo"]["weights"],
                         parameters["yolo"]["half"], parameters["yolo"]["img_size"],
                         parameters["carla"]["camera_width"], parameters["carla"]["camera_height"],
                         parameters["yolo"]["conf_thres"], parameters["yolo"]["iou_thres"],
                         parameters["yolo"]["max_det"])
        while not cam.ready:
            pass

        # Initialize vehicle control.
        vehicle_control: carla.VehicleControl = carla.VehicleControl()
        vehicle_control.manual_gear_shift = False

        # Initialize pedestrian control.
        pedestrian_control: carla.WalkerControl = carla.WalkerControl()
        pedestrian_control.speed = 1.5
        pedestrian_control.direction.x = 1
        pedestrian_control.direction.z = 0
        pedestrian.apply_control(pedestrian_control)

        # Attach network devices to pedestrian and vehicle, if we have V2P communication.
        if args.connected_mobility:
            env = NetworkEnvironment(display_manager, [1, 0], parameters["carla"]["vis_line_width"],
                                     parameters["carla"]["vis_trajectory_window_x"],
                                     parameters["carla"]["vis_trajectory_window_y"],
                                     parameters["carla"]["vis_point_radius"],
                                     parameters["carla"]["vis_scaling_factor_x"],
                                     parameters["carla"]["vis_scaling_factor_y"],
                                     parameters["carla"]["vis_sending_range"])

            ped_device = NetworkDevice(pedestrian, parameters["cm"]["max_range_pedestrian"], 1, 1,
                                       parameters["carla"]["scaling_factor"], parameters["cm"]["max_time_diff"])
            vehicle_device = NetworkDevice(ego_vehicle, parameters["cm"]["max_range_vehicle"], 2,
                                           parameters["carla"]["max_deceleration"],
                                           parameters["carla"]["scaling_factor"], parameters["cm"]["max_time_diff"])
            env.add_device(ped_device)
            env.add_device(vehicle_device)

        # Simulation loop.
        call_exit = False
        braking = False
        while True:
            if sync:
                world.tick()
            else:
                world.wait_for_tick()
            display_manager.render()

            # Network environment iteration.
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
                            vehicle_control.brake = 1.0
                            vehicle_control.throttle = 0.0

            if args.connected_mobility:
                # TODO.
                if vehicle_device.deceleration > 0.005:
                    braking = True
                    vehicle_control.brake = vehicle_device.deceleration
                    vehicle_control.throttle = 0.0

            if ego_vehicle.get_location().x < -40.0:
                vehicle_control.brake = 1.0
                vehicle_control.throttle = 0.0
            ego_vehicle.apply_control(vehicle_control)

            if args.connected_mobility and ped_device.deceleration > 0.0:
                pedestrian_control.speed = 0.0
            elif velocity < 0.1 and braking:
                pedestrian_control.direction.x = 0
                pedestrian_control.direction.y = 0
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


if __name__ == '__main__':
    with open("config.yaml", "r") as cfg_file:
        argparser = argparse.ArgumentParser(description="CARLA Autonomous Driving vs Connected Mobility")
        argparser.add_argument("--connected_mobility", action="store_true", help="Enable connected mobility")
        args = argparser.parse_args()
        parameters = yaml.safe_load(cfg_file)

        client = carla.Client(parameters["carla"]["host"], parameters["carla"]["port"])
        client.set_timeout(parameters["carla"]["timeout"])
        run_simulation(client, parameters)
