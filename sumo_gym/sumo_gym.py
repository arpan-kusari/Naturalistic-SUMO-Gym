# SUMOGym, Simulation Based AV Testing and Validation Package
# Copyright (C) 2021 University of Michigan Transportation Research Institute

import os
import sys

# check if SUMO_HOME exists in environment variable
# if not, then need to declare the variable before proceeding
# makes it OS-agnostic
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import gym
import math
import traci
import ast
import numpy as np
import pandas as pd
from typing import Tuple
from sumolib import checkBinary
from shapely.geometry import LineString, Point
import traci.constants as tc
from typing import List
import os
import random

Observation = np.ndarray
Action = np.ndarray


class SumoGym(gym.Env):
    """
    SUMO Environment for testing AV pipeline

    Uses OpenAI Gym
    @Params: scenario: choosing scenario type - "highway", "urban", "custom" or default
    @Params: choice: choosing particular scenario number of scenario type or "random" - chooses random scenario number
    @Params: delta_t: time-step of running simulation
    @Params: render_flag: whether to utilize SUMO-GUI or SUMO

    returns None
    """

    def __init__(self, scenario, choice, delta_t, render_flag=True) -> None:
        self.sumoBinary = None
        self.choice = choice
        self.delta_t = delta_t
        self.vehID = None
        self.egoID = None
        # Render the simulation
        self.render(render_flag)
        # User input scenarios
        self.scenario = scenario
        print("Running simulation in", scenario, " mode.")
        self._cfg = None

    def reset(self) -> Observation:
        """
        Function to reset the simulation and return the observation
        """
        if self.scenario == "urban":
            all_config_files = []
            config_dir = './urban_route_files'
            for file in os.listdir(config_dir):
                if file.endswith('.sumocfg'):
                    all_config_files.append(file)
            if self.choice == 'random':
                # choose a random sumo config file and run it
                chosen_file = random.choice(all_config_files)
            else:
                # check if the particular file is available
                # we have numbered the config files from 0
                # so just check if that number is lesser than/equal to available
                choice = int(self.choice)
                if choice <= len(all_config_files):
                    chosen_file = all_config_files[choice]
                else:
                    sys.exit('Chosen file is not available')
            self._cfg = config_dir + '/' + chosen_file
        elif self.scenario == "highway":
            all_config_files = []
            config_dir = './highway_route_files'
            for file in os.listdir(config_dir):
                if file.endswith('.sumocfg'):
                    all_config_files.append(file)
            all_config_files = sorted(all_config_files, key=len)
            if self.choice == 'random':
                chosen_file = random.choice(all_config_files)
            else:
                # check if the particular file is available
                # we have numbered the config files from 0
                # so just check if that number is lesser than/equal to available
                choice = int(self.choice)
                if choice < len(all_config_files):
                    chosen_file = all_config_files[choice]
                else:
                    sys.exit('Chosen file is not available')
            self._cfg = config_dir + '/' + chosen_file
        elif self.scenario == "custom":
            self._cfg = input("Please enter your custom .sumocfg filename:\n")
        else:
            # default
            self._cfg = "loop.sumocfg"
        # Start SUMO with the following arguments:
        # given config file
        # step length corresponding to the time step
        # warning during collision
        # minimum gap of zero
        # random placement as true
        sumoCmd = [
            self.sumoBinary,
            "-c",
            self._cfg,
            "--step-length",
            str(self.delta_t),
            "--collision.action",
            "warn",
            "--collision.mingap-factor",
            "0",
            "--random",
            "true",
        ]
        traci.start(sumoCmd)
        # run a single step
        traci.simulationStep()
        # get the vehicle id list
        self.vehID = traci.vehicle.getIDList()
        # select the ego vehicle ID
        self.egoID = input(
            "Please choose your ego vehicle ID (pressing enter only chooses 1st as ego): "
        )
        if self.egoID == "":
            self.egoID = self.vehID[0]
        # traci.gui.trackVehicle(traci.gui.DEFAULT_VIEW, self.egoID)
        # traci.gui.setZoom(traci.gui.DEFAULT_VIEW, 5000)
        # get observations with respect to the ego-vehicle
        obs = self._compute_observations(self.egoID)
        return obs

    def _get_features(self, vehID) -> np.ndarray:
        """
        Function to get the position, velocity and length of each vehicle
        """
        presence = 1
        x = traci.vehicle.getLanePosition(vehID)
        # right is negative and left is positiive
        y = traci.vehicle.getLateralLanePosition(vehID)
        lane_id = traci.vehicle.getLaneID(vehID)
        if lane_id == "":
            return None
        else:
            lane_index = traci.vehicle.getLaneIndex(vehID)
            lane_width = traci.lane.getWidth(lane_id)
        y = lane_width * (lane_index + 0.5) + y
        vx = traci.vehicle.getSpeed(vehID)
        vy = traci.vehicle.getLateralSpeed(vehID)
        length = traci.vehicle.getLength(vehID)
        distance_to_signal, signal_status, remaining_time = self._get_upcoming_signal_information(vehID)
        features = np.array([presence, x, y, vx, vy, length, distance_to_signal, signal_status, remaining_time])
        
        return features

    def _get_neighbor_ids(self, vehID) -> List:
        """
        Function to extract the ids of the neighbors of a given vehicle
        """
        neighbor_ids = []
        rightFollower = traci.vehicle.getRightFollowers(vehID)
        rightLeader = traci.vehicle.getRightLeaders(vehID)
        leftFollower = traci.vehicle.getLeftFollowers(vehID)
        leftLeader = traci.vehicle.getLeftLeaders(vehID)
        leader = traci.vehicle.getLeader(vehID)
        follower = traci.vehicle.getFollower(vehID)
        if len(leftLeader) != 0:
            neighbor_ids.append(leftLeader[0][0])
        else:
            neighbor_ids.append("")
        if leader is not None:
            neighbor_ids.append(leader[0])
        else:
            neighbor_ids.append("")
        if len(rightLeader) != 0:
            neighbor_ids.append(rightLeader[0][0])
        else:
            neighbor_ids.append("")
        if len(leftFollower) != 0:
            neighbor_ids.append(leftFollower[0][0])
        else:
            neighbor_ids.append("")
        if follower is not None and follower[0] != "":
            neighbor_ids.append(follower[0])
        else:
            neighbor_ids.append("")
        if len(rightFollower) != 0:
            neighbor_ids.append(rightFollower[0][0])
        else:
            neighbor_ids.append("")
        return neighbor_ids
   
    def _get_upcoming_signal_information(self, vehID):

        signal_information = traci.vehicle.getNextTLS(vehID)
        if len(signal_information) > 0:
            signal_id = signal_information[0][0]
            distance_to_signal = signal_information[0][2]
            signal_status = signal_information[0][3]
            remaining_time = traci.trafficlight.getNextSwitch(signal_id)-traci.simulation.getTime()

            if signal_status in ["G", "g"]:
                signal_status = 0
            elif signal_status in ["R", "r"]:
                signal_status = 1
            else:
                signal_status = 2
        else:
            distance_to_signal, signal_status, remaining_time = 0, 0, 0
        
        return distance_to_signal, signal_status, remaining_time
        
    def _compute_observations(self, vehID) -> Observation:
        """
        Function to compute the observation space
        Returns: A 7x9 array of Observations
        Key:
        Row 0 - ego and so on
        Columns:
        0 - presence
        1 - x in longitudinal lane position
        2 - y in lateral lane position
        3 - vx
        4 - vy
        5 - vehicle length
        6 - distance to next signal
        7 - current signal status, 0: green, 1: red, 2: yellow
        8 - remaning time of the current signal status in seconds
        """
        ego_features = self._get_features(vehID)

        neighbor_ids = self._get_neighbor_ids(vehID)
        obs = np.ndarray((7, 9))
        obs[0, :] = ego_features
        for i, neighbor_id in enumerate(neighbor_ids):
            if neighbor_id != "":
                features = self._get_features(neighbor_id)
                obs[i + 1, :] = features
            else:
                obs[i + 1, :] = np.zeros((9, ))

        return obs

    def long_lat_pos_cal(self, angle, acc_y, distance, heading):
        if angle <= 90:
            alpha = 90 - angle
            # consider steering maneuver
            if acc_y <= 0:
                radians = math.radians(alpha) - heading
            else:
                radians = math.radians(alpha) + heading
            dx = distance * math.cos(radians)
            dy = distance * math.sin(radians)
        elif 90 < angle <= 180:
            alpha = angle - 90
            # consider steering maneuver
            if acc_y <= 0:
                radians = math.radians(alpha) - heading
            else:
                radians = math.radians(alpha) + heading

            dx = distance * math.cos(radians)
            dy = -distance * math.sin(radians)
        elif 180 < angle <= 270:
            alpha = 270 - angle
            # consider steering maneuver
            if acc_y <= 0:
                radians = math.radians(alpha) + heading
            else:
                radians = math.radians(alpha) - heading

            dx = -distance * math.cos(radians)
            dy = -distance * math.sin(radians)
        else:
            alpha = angle - 270
            # consider steering maneuver
            if acc_y <= 0:
                radians = math.radians(alpha) + heading
            else:
                radians = math.radians(alpha) - heading

            dx = -distance * math.cos(radians)
            dy = distance * math.sin(radians)
        
        return dx, dy

    def update_state(self, action: Action) -> Tuple[float, float, float]:
        """
        Function to update the state of the ego vehicle based on the action (Accleration)
        Returns difference in position and current speed
        """
        # Longitudinal Acceleration Delay Parameter
        tc_ilc_long = 0.120
        # Lateral Acceleration Delay Parameter
        tc_ilc_lat = 0.120

        # pos = traci.vehicle.getPosition(self.egoID)
        acc = traci.vehicle.getAcceleration(self.egoID)
        angle = traci.vehicle.getAngle(self.egoID)
        lane_id = traci.vehicle.getLaneID(self.egoID)
        lane_shape = traci.lane.getShape(lane_id)
        curr_pos = traci.vehicle.getPosition(self.egoID)
        x, y = curr_pos[0], curr_pos[1]
        road_pos = traci.simulation.convertRoad(x, y)
        # Acceleration and Velocity in X and Y
        ax_cmd = action[0]
        ay_cmd = action[1]
        vx = traci.vehicle.getSpeed(self.egoID)  # long speed
        vy = traci.vehicle.getLateralSpeed(self.egoID)  # lat speed
        speed = math.sqrt(vx ** 2 + vy ** 2)
        # return heading in radians
        heading = math.atan(vy / (vx + 1e-12))
        acc_x = acc * math.cos(heading)
        acc_y = acc * math.sin(heading)

        acc_x += 1 / tc_ilc_long * (ax_cmd - acc_x) * self.delta_t
        acc_y += 1 / tc_ilc_lat * (ay_cmd - acc_y) * self.delta_t

        vx += acc_x * self.delta_t
        vy += acc_y * self.delta_t

        # stop the vehicle if speed is negative
        vx = max(0, vx)
        vy = max(0, vy)

        speed = math.sqrt(vx ** 2 + vy ** 2)
        distance = speed * self.delta_t
        long_distance = vx * self.delta_t
        lat_distance = vy * self.delta_t
        
        try:
            x_1, y_1 = traci.simulation.convert2D(road_pos[0], road_pos[1]+long_distance, road_pos[2])
            dx = x_1 - x
            dy = y_1 - y
            if angle <= 90:
                if acc_y <= 0:
                    dx -= lat_distance * math.cos(math.radians(angle))
                    dy += lat_distance * math.sin(math.radians(angle))
                else:
                    dx += lat_distance * math.cos(math.radians(angle))
                    dy -= lat_distance * math.sin(math.radians(angle))
            elif 90 < angle <= 180:
                angle = 180 - angle
                if acc_y <= 0:
                    dx += lat_distance * math.cos(math.radians(angle))
                    dy += lat_distance * math.sin(math.radians(angle))
                else:
                    dx -= lat_distance * math.cos(math.radians(angle))
                    dy -= lat_distance * math.sin(math.radians(angle))
            elif 180 < angle <= 270:
                angle = angle - 180
                if acc_y <= 0:
                    dx += lat_distance * math.cos(math.radians(angle))
                    dy -= lat_distance * math.sin(math.radians(angle))
                else:
                    dx -= lat_distance * math.cos(math.radians(angle))
                    dy += lat_distance * math.sin(math.radians(angle))
            else:
                angle = 360 - angle
                if acc_y <= 0:
                    dx -= lat_distance * math.cos(math.radians(angle))
                    dy -= lat_distance * math.sin(math.radians(angle))
                else:
                    dx += lat_distance * math.cos(math.radians(angle))
                    dy += lat_distance * math.sin(math.radians(angle))        
        except:
            dx, dy = self.long_lat_pos_cal(angle, acc_y, distance, heading)            
        
        return dx, dy, speed

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Function to take a single step in the simulation based on action for the ego-vehicle
        """
        # next state, reward, done (bool), info (dict)
        # dict --> useful info in event of crash or out-of-network
        # bool --> false default, true when finishes episode/sims
        # float --> reward = user defined func -- Zero for now (Compute reward functionality)
        curr_pos = traci.vehicle.getPosition(self.egoID)
        (dx, dy, speed) = self.update_state(action)
        
        new_x = curr_pos[0] + dx
        new_y = curr_pos[1] + dy

        edge = traci.vehicle.getRoadID(self.egoID)
        lane = traci.vehicle.getLaneIndex(self.egoID)
        lane_id = traci.vehicle.getLaneID(self.egoID)

        sim_check = False
        obs = []
        info = {}

        if lane_id == "":
            info["debug"] = "Ego-vehicle is out of network"
            sim_check = True

        elif self.egoID in traci.simulation.getCollidingVehiclesIDList():
            info["debug"] = "A crash happened to the Ego-vehicle"
            sim_check = True

        else:
            # ego-vehicle is mapped to the exact position in the network by setting keepRoute to 2

            traci.vehicle.moveToXY(
                self.egoID, edge, lane, new_x, new_y, tc.INVALID_DOUBLE_VALUE, 7
            )

            # remove control from SUMO, may result in very large speed
            traci.vehicle.setSpeedMode(self.egoID, 0)
            # set the speed of ego-vehicle
            traci.vehicle.setSpeed(self.egoID, speed)
            traci.simulationStep()
            obs = self._compute_observations(self.egoID)

        reward = self.reward(action)
        return obs, reward, sim_check, info

    def render(self, flag) -> None:
        """
        Function to render SUMO environment - essentially choosing SUMO-GUI or SUMO
        """

        # check the path of sumo/sumo-gui
        if "SUMO_HOME" in os.environ:
            bin_path = os.path.join(os.environ["SUMO_HOME"], "bin")
            sys.path.append(bin_path)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        # using sumo or sumo-gui
        if flag:
            self.sumoBinary = checkBinary("sumo-gui")
        else:
            self.sumoBinary = checkBinary("sumo")

    # Reward will not be implemented, user has the option
    def reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.
        :param action: the last action performed
        :return: the reward
        """
        reward = 0
        return reward

    def close(self):
        """
        Function to close the simulation environment
        """
        traci.close()