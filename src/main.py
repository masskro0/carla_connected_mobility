#!/usr/bin/env python

import argparse
import math
import yaml

import carla
import pygame
from pygame.locals import K_ESCAPE, K_q

from network import NetworkDevice, NetworkEnvironment
from utils import DisplayManager, PedestrianCamera, YOLOCamera


def run_simulation(clnt, params, sync=True):
    # Display Manager organizes everything we would like to see in a separate window.
    if args.connected_mobility:
        vis_x = params["carla"]["vis_trajectory_window_y"]
        display_manager = DisplayManager(window_size=[params["carla"]["screen_width"] + vis_x,
                                                      params["carla"]["screen_height"]])
    else:
        display_manager = DisplayManager(window_size=[params["carla"]["screen_width"],
                                                      params["carla"]["screen_height"]])

    world = clnt.get_world()
    original_settings = world.get_settings()
    actor_list = []
    try:
        # Simulation settings.
        if sync:
            traffic_manager = clnt.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            # settings.fixed_delta_seconds = 0.02    # Everything is too fast.
            world.apply_settings(settings)

        # Set spectator viewpoint.
        spectator: carla.Actor = world.get_spectator()
        location = carla.Location(x=-14.856689, y=131.858612, z=7.222219)
        rotation = carla.Rotation(pitch=-31.648739, yaw=4.472778, roll=0.000018)
        transform = carla.Transform(location, rotation)
        spectator.set_transform(transform)

        # Spawn ego vehicle.
        vehicle_blueprint = world.get_blueprint_library().find(params["carla"]["vehicle_model"])
        ego_location = carla.Location(x=65, y=130.5, z=0.600000)
        ego_rotation = carla.Rotation(pitch=0.0, yaw=-179.494202, roll=0.0)
        ego_position = carla.Transform(ego_location, ego_rotation)
        ego_vehicle: carla.Vehicle = world.spawn_actor(vehicle_blueprint, ego_position)
        actor_list.append(ego_vehicle)

        # Spawn child.
        pedestrian_blueprint = world.get_blueprint_library().find(params["carla"]["pedestrian_model"])
        pedestrian_location = carla.Location(x=-10.0, y=125.0, z=0.6)
        pedestrian_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        pedestrian_position = carla.Transform(pedestrian_location, pedestrian_rotation)
        pedestrian: carla.Actor = world.spawn_actor(pedestrian_blueprint, pedestrian_position)
        actor_list.append(pedestrian)

        # Attach camera to vehicle and load YOLO model.
        cam = YOLOCamera(world, display_manager, carla.Transform(carla.Location(x=0.5, z=1.5)), ego_vehicle, 0,
                         params["yolo"]["device"], params["yolo"]["data"], params["yolo"]["weights"],
                         params["yolo"]["half"], params["yolo"]["img_size"], params["carla"]["vehicle_cam_width"],
                         params["carla"]["vehicle_cam_height"], params["yolo"]["conf_thres"],
                         params["yolo"]["iou_thres"], params["yolo"]["max_det"])
        while not cam.ready:
            pass

        ped_cam = PedestrianCamera(world, display_manager, carla.Transform(carla.Location(x=0.1, z=1.0)), pedestrian,
                                   1, params["carla"]["pedestrian_cam_width"],
                                   params["carla"]["pedestrian_cam_height"])

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
            env = NetworkEnvironment(display_manager, 2, params["carla"]["vis_line_width"],
                                     params["carla"]["vis_trajectory_window_x"],
                                     params["carla"]["vis_trajectory_window_y"], params["carla"]["vis_point_radius"],
                                     params["carla"]["vis_scaling_factor_x"], params["carla"]["vis_scaling_factor_y"],
                                     params["carla"]["vis_sending_range"])

            ped_device = NetworkDevice(pedestrian, params["cm"]["max_range_pedestrian"], 1, 1,
                                       params["carla"]["scaling_factor"], params["cm"]["max_time_diff"])
            vehicle_device = NetworkDevice(ego_vehicle, params["cm"]["max_range_vehicle"], 2,
                                           params["carla"]["max_deceleration"], params["carla"]["scaling_factor"],
                                           params["cm"]["max_time_diff"])
            env.add_device(ped_device)
            env.add_device(vehicle_device)

        display_manager.calc_offsets()

        # Simulation loop.
        call_exit = False
        braking = False
        switched_img = False
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

            # Keep a constant velocity of 50km/h. Could be improved with a PID controller.
            ego_velocity = ego_vehicle.get_velocity().length() * 3.6
            if ego_velocity < 50.0 and not braking:
                vehicle_control.throttle = 0.7
            else:
                vehicle_control.throttle = 0.0

            if not args.connected_mobility:
                for xyxy in cam.people_list:
                    on_the_road = False
                    min_x = params["yolo"]["x_diff"]
                    max_x = params["carla"]["vehicle_cam_width"] - min_x
                    max_y = params["carla"]["vehicle_cam_height"]
                    min_y = max_y - params["yolo"]["y_diff"]
                    person_coords = [xyxy[0].cpu(), xyxy[1].cpu(), xyxy[2].cpu(), xyxy[3].cpu()]
                    if person_coords[0] >= min_x and person_coords[1] >= min_y and person_coords[2] <= max_x \
                            and person_coords[3] <= max_y:
                        on_the_road = True

                    if on_the_road and not braking:
                        braking = True
                        vehicle_control.brake = 1.0
                        vehicle_control.throttle = 0.0

            if args.connected_mobility:
                # Do not brake immediately, only after it gets "critical".
                if vehicle_device.deceleration > params["cm"]["deceleration_threshold"]:
                    braking = True
                    vehicle_control.brake = vehicle_device.deceleration
                    vehicle_control.throttle = 0.0

            # Stop the vehicle after a certain point, otherwise it crashes against the tree.
            if ego_vehicle.get_location().x < -40.0:
                vehicle_control.brake = 1.0
                vehicle_control.throttle = 0.0
            ego_vehicle.apply_control(vehicle_control)

            if (args.connected_mobility and ped_device.deceleration > 0.0) or (ego_velocity < 0.1 and braking):
                # Make the pedestrian stop.
                pedestrian_control.speed = 0.0
                if not switched_img:
                    ped_cam.switch_img()
                    switched_img = True
            elif pedestrian.get_location().x >= -3.8:
                # Make the pedestrian cross the road near the parked car.
                pedestrian_control.direction.x = 0
                pedestrian_control.direction.y = 1
            pedestrian.apply_control(pedestrian_control)

            # Terminate code when hitting 'q' or 'ESC'.
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
        # Cleanup everything when the code terminates.
        if display_manager:
            display_manager.destroy()
        clnt.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        world.apply_settings(original_settings)


if __name__ == '__main__':
    # Get parameters from the config file.
    with open("config.yaml", "r") as cfg_file:
        argparser = argparse.ArgumentParser(description="CARLA Autonomous Driving vs Connected Mobility")
        argparser.add_argument("--connected_mobility", action="store_true", help="Enable connected mobility")
        args = argparser.parse_args()
        parameters = yaml.safe_load(cfg_file)

        # Create and connected the client to the Carla simulator.
        client = carla.Client(parameters["carla"]["host"], parameters["carla"]["port"])
        client.set_timeout(parameters["carla"]["timeout"])

        # Run this scenario.
        run_simulation(client, parameters)
