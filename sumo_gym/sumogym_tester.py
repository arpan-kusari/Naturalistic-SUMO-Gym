# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

# @author      Nikhil Punshi
# @supervisor  Arpan Kusari
# @date        08-16-2021

# SumoGym Testing File
from sumo_gym import SumoGym
import numpy as np

Observation = np.ndarray
Action = np.ndarray


# Create a simple IDM model based on default parameters for testing SumoGym
def simple_idm(obs, max_speed, max_acc_x) -> (float, float, float):
    v0 = max_speed
    s0 = 20
    T = 1.5
    a = 1.2
    b = 2
    delta = 4
    ego_x = obs[0, 1]
    ego_v = obs[0, 3]
    # have to check if lead_vehicle exists
    if obs[2, 0] == 1:
        lead_vehicle_v = obs[2, 3]
        lead_vehicle_x = obs[2, 1]
        lead_vehicle_length = obs[2, -1]
        del_x = lead_vehicle_x - ego_x - lead_vehicle_length
        del_v = lead_vehicle_v - ego_v
        # print('del_x = ', del_x, ' del_v = ', del_v)
    else:
        del_x = 150
        del_v = max_speed - ego_v

    # https://traffic-simulation.de/info/info_IDM.html
    d0 = s0 + max(0, ego_v * T + ego_v * del_v / (2 * np.sqrt(a * b)))
    acc_x = a * (1 - (ego_v / v0) ** delta - (d0 / del_x) ** 2)

    # acc_x = min(max(acc_x, -max_acc_x), max_acc_x)
    return acc_x, del_x, del_v


# Call SumoGym with the following parameters:
#   scenario: type of scenario the user would like to run (highway or urban)
#   choice: which particular scenario number does the user want to run - can be random or a specific numeric choice within quotes
#   delta_t: simulation time step, a small number will smooth the vehicle's trajectory
#   render_flag: Whether to render or to suppress
env = SumoGym(scenario='highway', choice='random', delta_t=0.01, render_flag=True)
# resets the simulation by dropping the vehicles into the environment and returning observations
obs = env.reset()
# user can run the simulation for a certain number of steps
num = input("Please enter simulation time (pressing enter only runs till done): ")
done = False
# maximum speed of ego vehicle
max_speed = 30
# maximum acceptable acceleration
max_acc_long = 3
# initialization of info dict that returns data in the event of crash or out-of-network
info = {}

# if user returns enter, then simulation will run indefinitely until the ego vehicle crashes or goes outside network
if num == "":
    iter = 0
    while done is False:
        # get the longitudinal acceleration, distance to front vehicle and relative velocity
        acc_long, del_x, del_v = simple_idm(obs, max_speed, max_acc_long)
        # The action to be sent to SumoGym is longitudinal and lateral acceleration
        Action = [acc_long, 2.]
        print("Iter: ", iter, " Long acc: ", acc_long)
        obs, reward, done, info = env.step(action=Action)
        iter += 1
else:
    for _ in range(int(num)):
        acc_long, del_x, del_v  = simple_idm(obs, max_speed, max_acc_long)
        Action = [acc_long, 0.]
        obs, reward, done, info = env.step(action=Action)

print("Info: ", info)
env.close()
