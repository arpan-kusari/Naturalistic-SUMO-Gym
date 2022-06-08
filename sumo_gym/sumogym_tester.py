# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

# @author      Nikhil Punshi
# @supervisor  Arpan Kusari
# @date        08-16-2021

# SumoGym Testing File

import os
import math
from sumo_gym import SumoGym
from params import IDMConstants, LCConstants
import numpy as np
import matplotlib.pyplot as plt
Observation = np.ndarray
Action = np.ndarray


# Create a simple IDM model based on default parameters for testing SumoGym
def simple_idm(obs, max_speed, max_acc_x) -> (float, float, float):
    idm_params = IDMConstants()
    v0 = max_speed
    ego_x = obs[0, 1]
    ego_v = obs[0, 3]
    # have to check if lead_vehicle exists
    if obs[2, 0] == 1:
        lead_vehicle_v = obs[2, 3]
        lead_vehicle_x = obs[2, 1]
        lead_vehicle_length = obs[2, 5]
        del_x = lead_vehicle_x - ego_x - lead_vehicle_length
        del_v = lead_vehicle_v - ego_v
        # print('del_x = ', del_x, ' del_v = ', del_v)
    else:
        del_x = 150
        del_v = max_speed - ego_v

    # https://traffic-simulation.de/info/info_IDM.html
    d0 = idm_params.s0 + max(0, ego_v * idm_params.T + ego_v * del_v /
                             (2 * np.sqrt(idm_params.a * idm_params.b)))
    acc_x = idm_params.a * (1 - (ego_v / v0) ** idm_params.delta - (d0 / del_x) ** 2)

    # acc_x = min(max(acc_x, -max_acc_x), max_acc_x)
    return acc_x, del_x, del_v


def simple_lane_change(obs,
                       sampling_time,
                       lc_bool,
                       curr_lc_time,
                       desired_lane,
                       max_ind,
                       num_lanes,
                       lane_width) -> (float, bool, np.ndarray):
    lc_params = LCConstants()
    if lc_bool is False:
        curr_lane = obs[0, 5]
        # find random probability of being in a lane - [left, curr, right]
        prob = np.random.rand(1, 3)
        max_ind = np.argmax(prob)
        if ((max_ind == 0 and curr_lane == num_lanes-1)
                or max_ind == 1
                or (max_ind == 2 and curr_lane == 0)):
            return 0, lc_bool, curr_lane, curr_lc_time, max_ind
        else:
            lc_bool = True
            # max_ind is indexed from 0 to 2
            # lanes should be indexed from -1 to 1
            # right most lane is 0 and left most lane is 2
            desired_lane = curr_lane - (max_ind - 1)
            print("Curr lane = ", curr_lane, "Desired lane = ", desired_lane, "Max ind = ", max_ind)
    if lc_bool:
        curr_lc_time += sampling_time
        y = obs[0, 2]
        if curr_lc_time > lc_params.time_for_lane_change:
            lc_bool = False
            curr_lc_time = 0
            desired_lane = obs[0, 5]
            return 0, lc_bool, desired_lane, curr_lc_time, max_ind
        vy = obs[0, 4]
        ay_max = (2 * math.pi * lane_width)/(lc_params.time_for_lane_change**2)
        ay_cmd = ay_max * math.sin((2*math.pi)*curr_lc_time/lc_params.time_for_lane_change)
        if max_ind == 2:
            ay_cmd = -ay_cmd
        print('ay: ', ay_cmd, ' vy: ', vy, ' y:', y, 'lat_dist', vy*delta_t, 'del_y:', abs((desired_lane + 0.5) * lane_width - y))
        return ay_cmd, lc_bool, desired_lane, curr_lc_time, max_ind


# Call SumoGym with the following parameters:
#   scenario: type of scenario the user would like to run (highway or urban)
#   choice: which particular scenario number does the user want to run - can be random or a specific numeric choice within quotes
#   delta_t: simulation time step, a small number will smooth the vehicle's trajectory
#   render_flag: Whether to render or to suppress
delta_t = 0.1
env = SumoGym(scenario='highway', choice='random', delta_t=delta_t, render_flag=True)
# resets the simulation by dropping the vehicles into the environment and returning observations
obs = env.reset()
# user can run the simulation for a certain number of steps
num = input("Please enter simulation time (pressing enter only runs till done): ")
done = False
# maximum speed of ego vehicle
max_speed = 40
# maximum acceptable acceleration
max_acc_long = 3
# initialization of info dict that returns data in the event of crash or out-of-network
info = {}
# plt.figure()
# if user returns enter, then simulation will run indefinitely until the ego vehicle crashes or goes outside network
desired_lane = obs[0, 5]
lc_bool = False
curr_lc_time = 0
max_ind = 1
lat_dist = 0
if num == "":
    iter = 0
    while done is False:
        # get the longitudinal acceleration, distance to front vehicle and relative velocity
        acc_long, del_x, del_v = simple_idm(obs, max_speed, max_acc_long)
        # get lateral acceleration
        [acc_lat,
         lc_bool,
         desired_lane,
         curr_lc_time,
         max_ind] = simple_lane_change(obs=obs,
                                       sampling_time=delta_t,
                                       lc_bool=lc_bool,
                                       curr_lc_time=curr_lc_time,
                                       desired_lane=desired_lane,
                                       max_ind=max_ind,
                                       num_lanes=env.get_num_lanes(),
                                       lane_width=env.get_lane_width())
        # The action to be sent to SumoGym is longitudinal and lateral acceleration
        Action = [acc_long, acc_lat]
        # print("Iter: ", iter, " Long acc: ", acc_long)
        obs, reward, done, info = env.step(action=Action)
        iter += 1
else:
    for _ in range(int(num)):
        acc_long, del_x, del_v  = simple_idm(obs, max_speed, max_acc_long)
        Action = [acc_long, 20]
        obs, reward, done, info = env.step(action=Action)
env.close()
