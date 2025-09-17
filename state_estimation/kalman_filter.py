import numpy as np

# Blimp state
#   --- CARTESIAN STATE ---
#   3: cartesian positions in XYZ
#   3: cartesian velocities in XYZ
#
#   --- QUATERNIONS ---
#   angles but without the trigo shitshow
#   4: quaternions
#
#   --- BIASES ---
#   Only add the temporaly changing biases (ie ignore LIDAR, magnetometer or baro long term drift)
#   3: accel_1 biaises
#   3: accel_2 biaises
#   3: gyro_1 biaises
#   3: gyro_2 biaises

T = 2
pos = np.zeros([3, T])
vel = np.zeros_like(pos)
quat = np.zeros([4, T])
ac1_b = np.zeros_like(pos)
ac2_b = np.zeros_like(pos)
gy1_b = np.zeros_like(pos)
gy2_b = np.zeros_like(pos)


state = np.vstack([pos, vel, quat, ac1_b, ac2_b, gy1_b, gy2_b])

print(state)


def compute_F(state, command):
    pass

def predict_next_state(state, command, K):
    """ predict next state given previous state, applied command and predicted covar"""
    pass

def predict_P(P, F):
    """ update state covariance """


