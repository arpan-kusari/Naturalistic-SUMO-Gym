class SimulationConstants:
    def __init__(self):
        self.num_features = 10
        '''
        Action type: choose between the following action types
        acceleration:
            Action = [acceleration_x, acceleration_y]
        acc_steering:
            Action = [acceleration, steering]
            update according to bicycle model, where the reference point is center of gravity
        '''
        self.action_type = "acceleration"


class IDMConstants:
    def __init__(self):
        self.s0 = 20
        self.T = 1.5
        self.a = 1.2
        self.b = 2
        self.delta = 4


class LCConstants:
    def __init__(self):
        self.zeta_y = 0.9
        self.time_for_lane_change = 5
        self.wn_y = 4 / self.time_for_lane_change / self.zeta_y
        self.K1y = 2 * self.wn_y * self.zeta_y
        self.K2y = self.wn_y * self.wn_y
        self.ay_max = 2.0
        self.ay_dot_max = 4
