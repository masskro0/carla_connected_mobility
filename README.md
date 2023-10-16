# Carla Connected Mobility vs Autonomous Driving
A simple scenario to demonstrate where an autonomous vehicle cannot avoid a crash, while via vehicle-to-pedestrian (V2P) communication the crash can be avoided.

This scenario is made for demonstration purposes only.

## Autonomous Driving Mode
https://github.com/masskro0/carla_connected_mobility/assets/49821640/02e99d7f-c8cd-4c20-af03-fa35d3de3a4b



By using autonomous driving only, the vehicle cannot avoid the crash.
<br><br>

## Connected Mobility Mode
https://github.com/masskro0/carla_connected_mobility/assets/49821640/34e05754-ce0b-4cd3-836c-11fd97f113db

By using additional network devices with vehicle-to-pedestrian communication (V2P), the vehicle receives the trajectory of the pedestrian, determines an intersection of both and reacts much earlier than in the first scenario. The crash is avoided in this scenario.

## Setup
- Carla 0.9.13
- Ubuntu 20.04
- Python 3.8.10

## Installation
Run `pip install -r requirements.txt`

## Launching
Inside the src folder, run `python3 main.py` for the autonommous driving mode. 
Add the flag `--connected_mobility` for V2P communication.
