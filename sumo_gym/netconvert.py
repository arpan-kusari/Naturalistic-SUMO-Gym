import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from shapely.geometry import LineString, Point
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

lane_ids = []
lane_shapes = []
lane_shape_x = []
lane_shape_y = []

net = sumolib.net.readNet(r'./highway_route_files/highway-100.net.xml')
# net = sumolib.net.readNet(r'./loop.net.xml')
for edge in net.getEdges():
    for lane in edge.getLanes():
        lane_id = lane.getID()
        lane_shape = lane.getShape()
        x_list = [x[0] for x in lane_shape]
        y_list = [x[1] for x in lane_shape]
        coords = [(x, y) for x,y in zip(x_list, y_list)]
        line = LineString(coords)
        line_length = line.length
        x_list_new = []
        y_list_new = []
        # for i in range(1, int(line_length)+1, 1):
        # if lane_id == '124433726_0':
            # print(line_length)
        for i in np.linspace(0, line_length, 3*int(line_length)):
            # print(i)
            new_point = line.interpolate(i)
            x_list_new.append(new_point.coords[0][0])
            y_list_new.append(new_point.coords[0][1])
        # print(line_length)
        coords_new = [(x, y) for x,y in zip(x_list_new, y_list_new)]

        lane_ids.append(lane_id)
        lane_shape_x.append(x_list_new)
        lane_shape_y.append(y_list_new)
        
df = pd.DataFrame({'lane_id':lane_ids,
                   'x':lane_shape_x,
                   'y':lane_shape_y})
df.to_csv(r"./lane_shape_info.csv", index=False)

# df = pd.read_csv(r"./lane_shape_info_loop.csv")

# x = ast.literal_eval(df[df['lane_id']=='124433726_0']['x'].values[0])
# y = ast.literal_eval(df[df['lane_id']=='124433726_0']['y'].values[0])

# plt.figure()
# plt.scatter(x, y)
# plt.show()