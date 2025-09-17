import numpy as np
from scipy.spatial.transform import Rotation as R

# Constants (example values, adjust to your blimp)
MASS = 1.5          # kg
GRAVITY = np.array([0, 0, -9.81])  # m/s^2 in WORLD frame

# State vector layout indices
IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_QUAT = slice(6, 10)
IDX_AC1_B = slice(10, 13)
IDX_AC2_B = slice(13, 16)
IDX_GY1_B = slice(16, 19)
IDX_GY2_B = slice(19, 22)
IDX_OMEGA = slice(22, 25)
STATE_SIZE = 25

MOTOR_DIRS = np.array([[1, 1, 0],
                       [0, 0, 1],
                       [1, -1, 0],
                       [0, 0, 1],
                       [-1, -1, 0],
                       [0, 0, 1],
                       [-1, 1, 0],
                       [0, 0, 1]])

MOTOR_POS = np.array([
    [ 0.5,  0.5, 0.0],
    [ 0.5,  0.5, 0.0],
    [ 0.5, -0.5, 0.0],
    [ 0.5, -0.5, 0.0],
    [-0.5, -0.5, 0.0],
    [-0.5, -0.5, 0.0],
    [-0.5,  0.5, 0.0],
    [-0.5,  0.5, 0.0],
])

# 8 anchors at known (x,y,z) in the hangar (meters)
ANCHOR_POS = np.array([
    [ 0.0,   0.0,   2.0],
    [50.0,   0.0,   2.0],
    [50.0,  50.0,   2.0],
    [ 0.0,  50.0,   2.0],
    [25.0,   0.0,   2.0],
    [50.0,  25.0,   2.0],
    [25.0,  50.0,   2.0],
    [ 0.0,  25.0,   2.0],
])  # shape (8,3)

# Blimp’s inertia tensor in BODY frame (assumed diagonal here)
Ixx, Iyy, Izz = 0.1, 0.12, 0.08  # adjust to your gondola
INERTIA = np.diag([Ixx, Iyy, Izz])

# Finite-difference epsilon for Jacobian
EPS = 1e-4

def construct_true_state(x, u, dt):
    # x is the previous true state, shape (25,)
    p        = x[IDX_POS]
    v        = x[IDX_VEL]
    q        = x[IDX_QUAT]
    acc1_b   = x[IDX_AC1_B]
    acc2_b   = x[IDX_AC2_B]
    gyro1_b  = x[IDX_GY1_B]
    gyro2_b  = x[IDX_GY2_B]
    omega    = x[IDX_OMEGA]        # <-- pull last true omega

    # 1) Position
    p_pred = p + v * dt

    # 2) Velocity
    F_body  = MOTOR_DIRS.T @ u
    F_world = R.from_quat(q).apply(F_body)
    a_world = F_world/MASS + GRAVITY
    v_pred  = v + a_world*dt

    # 3) Rotational dynamics
    forces   = MOTOR_DIRS * u[:,None]
    torques  = np.cross(MOTOR_POS, forces)
    tau_body = torques.sum(axis=0)

    omega_dot  = np.linalg.inv(INERTIA) @ (tau_body - np.cross(omega, INERTIA@omega))
    omega_pred = omega + omega_dot*dt

    # 4) Quaternion
    dq_rot = R.from_rotvec(omega_pred * dt)
    q_pred = (R.from_quat(q) * dq_rot).as_quat()
    qn = np.linalg.norm(q_pred)
    if qn < 1e-8:
        q_pred = np.array([1.,0.,0.,0.])
    else:
        q_pred /= qn

    # 5) Biases (random walk zero-mean)
    acc1_b_pred  = acc1_b
    acc2_b_pred  = acc2_b
    gyro1_b_pred = gyro1_b
    gyro2_b_pred = gyro2_b

    # 6) Pack new true state
    x_true = np.zeros_like(x)
    x_true[IDX_POS]    = p_pred
    x_true[IDX_VEL]    = v_pred
    x_true[IDX_QUAT]   = q_pred
    x_true[IDX_OMEGA]  = omega_pred
    x_true[IDX_AC1_B]  = acc1_b_pred
    x_true[IDX_AC2_B]  = acc2_b_pred
    x_true[IDX_GY1_B]  = gyro1_b_pred
    x_true[IDX_GY2_B]  = gyro2_b_pred

    return x_true


def predict_next_state(x, u, dt):
    """
    Propagate state x (22×1) with command u (8×1 thrusts) over dt.
    Returns x_pred (22×1).
    """
    # Decompose state
    p = x[IDX_POS]
    v = x[IDX_VEL]
    q = x[IDX_QUAT]
    acc1_b = x[IDX_AC1_B]
    acc2_b = x[IDX_AC2_B]
    gyro1_b = x[IDX_GY1_B]
    gyro2_b = x[IDX_GY2_B]
    measured_omega = x[IDX_OMEGA]

    # 1) Position update
    p_pred = p + v * dt

    # 2) Velocity update: sum thrusts -> force in body -> rotate to world
    # total force in body frame
    F_body = MOTOR_DIRS.T @ u  # shape (3,)
    # rotate to world
    rot = R.from_quat(q)
    F_world = rot.apply(F_body)
    a_world = F_world / MASS + GRAVITY
    v_pred = v + a_world * dt

    forces    = MOTOR_DIRS * u[:,None]             # (8×3)
    torques   = np.cross(MOTOR_POS, forces)        # (8×3) each r_i × F_i
    tau_body  = torques.sum(axis=0)                # total torque in BODY

    omega = measured_omega - 0.5*(gyro1_b + gyro2_b)   # shape (3,)

    # rotational dynamics: I * ω̇ + ω × (I ω) = τ
    omega_dot = np.linalg.inv(INERTIA) @ (tau_body - np.cross(omega, INERTIA @ omega))

    # integrate angular rate
    omega_pred = omega + omega_dot * dt

    # quaternion integration via small-angle
    dq_rot = R.from_rotvec((omega_pred - gyro1_b)*dt)
    q_pred  = (R.from_quat(q) * dq_rot).as_quat()
    q_norm = np.linalg.norm(q_pred)
    if q_norm < 1e-8:
       # fallback to identity (or previous quat)
       q_pred = np.array([1.0, 0.0, 0.0, 0.0])
       print("something something quaternion problem")
    else:
       q_pred /= q_norm

    # 4) Bias states: we model as constant (random walk elsewhere via Q)
    acc1_b_pred = acc1_b
    acc2_b_pred = acc2_b
    gyro1_b_pred = gyro1_b
    gyro2_b_pred = gyro2_b

    # stack into state vector
    x_pred = np.zeros_like(x)
    x_pred[IDX_POS] = p_pred
    x_pred[IDX_VEL] = v_pred
    x_pred[IDX_QUAT] = q_pred
    x_pred[IDX_AC1_B] = acc1_b_pred
    x_pred[IDX_AC2_B] = acc2_b_pred
    x_pred[IDX_GY1_B] = gyro1_b_pred
    x_pred[IDX_GY2_B] = gyro2_b_pred
    x_pred[IDX_OMEGA] = omega_pred
    return x_pred


def compute_F(x, u, dt):
    """
    Compute Jacobian F = df/dx at state x and command u via finite differences.
    Returns F (22×22).
    """
    F = np.zeros((STATE_SIZE, STATE_SIZE))
    f0 = predict_next_state(x, u, dt)
    for i in range(STATE_SIZE):
        dx = np.zeros_like(x)
        dx[i] = EPS
        f1 = predict_next_state(x + dx, u, dt)
        F[:, i] = (f1 - f0) / EPS
    return F

def compute_H_acc(x_pred, x_est, acc_index, dt):
    # unpack
    q_pred = x_pred[IDX_QUAT]         # shape (4,)
    v_pred = x_pred[IDX_VEL]          # shape (3,)
    v_prev = x_est[IDX_VEL]   # your previous-step velocity

    # body→world rotation
    rot = R.from_quat(q_pred)
    Rwb = rot.as_matrix()             # world from body
    Rbw = Rwb.T                       # body from world

    # delta‐v used in measurement model
    dv = (v_pred - v_prev) / dt

    # Allocate 3×22
    H_acc = np.zeros((3, STATE_SIZE))

    # ∂h/∂v  → R_bw / DT
    H_acc[:, 3:6] = Rbw / dt

    # ∂h/∂q  → (∂(R_bw·dv)/∂q), a 3×4 block
    # You can compute this analytically, or finite‐difference it:
    eps = 1e-6
    for i in range(4):
        dq = np.zeros(4); dq[i] = eps
        rp = R.from_quat(q_pred + dq).as_matrix().T @ dv
        rm = Rbw @ dv
        H_acc[:, 6+i] = (rp - rm) / eps

    # ∂h/∂b_a1  → identity
    if acc_index == 1:
        H_acc[:, IDX_AC1_B] = np.eye(3)
    if acc_index == 2:
        H_acc[:, IDX_AC2_B] = np.eye(3)
    return H_acc

def compute_H_gyro(gyro_index):
    H_gyro = np.zeros((3, STATE_SIZE))

    if gyro_index == 1:
        H_gyro[:, IDX_OMEGA] = np.eye(3)
        H_gyro[:, IDX_GY1_B]   = np.eye(3)
    elif gyro_index == 2:
        H_gyro[:, IDX_OMEGA] = np.eye(3)
        H_gyro[:, IDX_GY2_B] = np.eye(3)
    
    return H_gyro

def compute_H_anchor(x_pred):
    p = x_pred[IDX_POS]               # (3,)
    ranges = np.linalg.norm(p - ANCHOR_POS, axis=1)  # (8,)
    H = np.zeros((8, STATE_SIZE))
    for i in range(8):
        diff = p - ANCHOR_POS[i]
        H[i, 0:3] = diff / ranges[i]
    return H     # shape (8,25)

def command_generation(k):
    if k < 100:
        return np.array([ 0,10000, 0,10000, 0,10000, 0,10000])
    elif (100 <= k) and (k < 150):
        return np.array([0, 1, 0, 1, 0, 1, 0, 1])
    elif (150 <= k) and (k < 180):
        return np.array([0, 0, 0, 0, 0, 0, 0, 0])
    elif (180 <= k) and (k < 200):
        return np.array([-1, 0, 1, 0, -1, 0, 1, 0])
    else:
        return np.array([-1, -1, -1, -1, -1, -1, -1, -1])
    return np.array([1, 0.2, 0, 0.2, 0, 0.2, 1, 0.2])


def h_accel(x_mes, prev_x_mes, acc_index, dt):
    # world‐frame accel = (v_k – v_k-1)/dt – gravity
     # then rotate into body: a_body = R_bw @ (a_world)
     dv = (x_mes[IDX_VEL] - prev_x_mes[IDX_VEL]) / dt - GRAVITY
     q_k = x_mes[IDX_QUAT]
     Rbw = R.from_quat(q_k).as_matrix().T
     a_body = Rbw @ dv
     # subtract bias
     if acc_index == 1:
         return a_body - x_mes[IDX_AC1_B]
     else:
         return a_body - x_mes[IDX_AC2_B]


def h_gyro(x_mes, gyro_index):
    return x_mes[IDX_OMEGA] + x_mes[IDX_GY1_B]

def h_anchor(x_mes, anchor_index):
    p = x_mes[IDX_POS]               # (3,)
    # compute 8 slant‐ranges
    return np.linalg.norm(p - ANCHOR_POS, axis=1)  # (8,)

def h_combined(x_pred, x_prev, dt):
    # Accelerometer 1 prediction
    a1_pred = h_accel(x_pred, x_prev, 1, dt)
    # Gyro 1 prediction
    g1_pred = h_gyro (x_pred, gyro_index=1)
    # Accelerometer 2
    a2_pred = h_accel(x_pred, x_prev, 2, dt)
    # Gyro 2
    g2_pred = h_gyro (x_pred, gyro_index=2)
    # Barometer predicts p_z
    b_pred  = x_pred[IDX_POS][2]   # the z-component of position
    # anchor
    anchor_pred = h_anchor(x_pred, anchor_index = 0)
    # stack into one vector of length 21
    return np.hstack([a1_pred, g1_pred, a2_pred, g2_pred, b_pred, anchor_pred])


if __name__ == "__main__":

    T = 50
    t = np.linspace(0, 10, T)               # time vector
    dt = 0.01

    x_true = np.zeros([STATE_SIZE, T])
    x_est = np.zeros_like(x_true)
    x_pred = np.zeros_like(x_true)    

    

    # 13 : 3 + 3 + 3 + 3 + 1 + 8 -- 2 * IMU + 2 * gyro + 1 altimeter + 8 UWB anchors
    z_meas = np.zeros([21, T])
    z_pred = np.zeros_like(z_meas)
    z_error = np.zeros_like(z_meas)

    # set initial quaternion = identity
    for k in range(T):
        x_true[IDX_QUAT, k] = np.array([1.0, 0.0, 0.0, 0.0])
        x_est [IDX_QUAT, k] = np.array([1.0, 0.0, 0.0, 0.0])


    # Command using motors
    command = np.zeros((8,  T))

    for k in range(T):
        # e.g. u[:,k] = mix_altitude_climb(desired_alt[k], current_altitude) 
        command[:, k] = command_generation(k)


    #   0. Initialization
    # 0.1 Process noise covariance Q_k (22×22)
    #   – process noise on position (from accel‐noise integration)
    #   – process noise on velocity (from accel noise)
    #   – process noise on quaternion (from gyro noise)
    #   – random‐walk noise on accel biases and gyro biases
    q_pos   = 1e-6    # m^2
    q_vel   = 1e-4    # (m/s)^2
    q_quat  = 1e-4    # (unitless, quaternion error)
    q_ba    = 1e-8    # (m/s²)^2 per step
    q_bw    = 1e-6   # (rad/s)^2 per step

    Q_k = np.diag(np.concatenate([
        np.ones(3)*q_pos,
        np.ones(3)*q_vel,
        np.ones(4)*q_quat,
        np.ones(3)*q_ba,     # accel‐1 bias
        np.ones(3)*q_ba,     # accel‐2 bias
        np.ones(3)*q_bw,     # gyro‐1 bias
        np.ones(3)*q_bw,      # gyro‐2 bias
        np.ones(3)
    ]))

    # 0.2 Measurement noise covariance R_k
    #   Measurement vector z = [ acc1(3), gyro1(3), acc2(3), gyro2(3), baro(1) ]
    #   – accel noise: 40 µg → sig_acc ≈ 40e-6·9.81 m/s²
    #   – gyro noise: 0.1 °/s → sig_gyro ≈ 0.1·π/180 rad/s
    #   – baro noise: 0.3 m
    sig_acc  = 40e-6 * 9.81
    sig_gyro = np.deg2rad(0.1)
    sig_baro = 0.3
    sig_anchor = 0.1

    R_k = np.diag(np.concatenate([
        np.ones(3)*(sig_acc**2),
        np.ones(3)*((sig_gyro+0.005)**2),
        np.ones(3)*((sig_acc+0.005)**2),
        np.ones(3)*(sig_gyro**2),
        [sig_baro**2],
        np.ones(8)* (sig_anchor**2)
    ]))

    """ rank = np.linalg.matrix_rank(R_k)
    print("R rank:", rank, "/ 13") """

    # 0.3 Initial state covariance P_k
    #   – position uncertain to ±1 m
    #   – velocity uncertain to ±1 m/s
    #   – attitude uncertain to ±5°
    #   – biases uncertain to ±0.1 m/s² and ±0.01 rad/s
    p0_var      = 1.0
    v0_var      = 1.0
    quat0_var   = np.deg2rad(5.0)**2
    ba0_var     = (0.1)**2
    bw0_var     = (0.01)**2

    P_k = np.diag(np.concatenate([
        np.ones(3)*p0_var,
        np.ones(3)*v0_var,
        np.ones(4)*quat0_var,
        np.ones(3)*ba0_var,
        np.ones(3)*ba0_var,
        np.ones(3)*bw0_var,
        np.ones(3)*bw0_var,
        np.ones(3)
    ]))


    for k in range(1, T):
        print(f"iter {k}")
        #   0. Fill the ground truth
        x_true[:, k] = construct_true_state(x_true[:, k-1], command[:, k-1], dt)

        #   1. Prediction
        #   1.1. State prediction
        x_pred[:, k] = predict_next_state(x_est[:, k-1], command[:, k-1], dt)

        #   1.2. Update the state model function
        F_k    = compute_F(x_est[:,k-1], command[:,k-1], dt)

        #   1.2. State covariance update
        P_k = F_k @ P_k @ F_k.T + Q_k

        H_acc1 = compute_H_acc(x_pred[:, k], x_est[:, k-1], 1, dt)
        H_acc2 = compute_H_acc(x_pred[:, k], x_est[:, k-1], 2, dt)

        H_gyro1 = compute_H_gyro(1)
        H_gyro2 = compute_H_gyro(2)

        H_baro       = np.zeros((1, STATE_SIZE))
        H_baro[0, 2] = 1.0   # ∂h_baro/∂p_z = 1

        H_anchor = compute_H_anchor(x_pred[:, k])

        # GENERATE VIRTUAL DATA WITH NOICE RELATIVE TO THE COMPOSANTS INDICATED STD

        # — 3.2 Measurement generation —
        #  * Accelerometer #1
        acc1_meas = h_accel( x_true[:,k], x_true[:,k-1], 1, dt) + np.random.randn(3)*sig_acc
        #  * Gyro #1
        gyro1_meas = h_gyro( x_true[:,k], gyro_index=1) + np.random.randn(3)*sig_gyro
        #  * Accelerometer #2
        acc2_meas = h_accel( x_true[:,k], x_true[:,k-1], 2, dt) + np.random.randn(3)*sig_acc
        #  * Gyro #2
        gyro2_meas = h_gyro (x_true[:,k], gyro_index=2) + np.random.randn(3)*sig_gyro
        #  * Barometer
        baro_meas = x_true[2,k] + np.random.randn()*sig_baro
        #   * UWB
        anchor_meas = np.linalg.norm(x_true[IDX_POS, k] - ANCHOR_POS, axis=1) + np.random.randn(8)*sig_anchor
        

        z_meas[:, k] = np.hstack([acc1_meas, gyro1_meas, acc2_meas, gyro2_meas, baro_meas, anchor_meas])

        H_k   = np.vstack([H_acc1, H_gyro1, H_acc2, H_gyro2, H_baro, H_anchor])
        """ rank = np.linalg.matrix_rank(H_k)
        print("H rank:", rank, "/ 13") """

        #   2. Correction given current measured state
        #   2.1. Error
        z_pred[:, k] = h_combined(x_pred[:, k], x_est[:, k-1], dt)
        z_error[:, k] = z_meas[:, k] - z_pred[:, k]
        #   2.2. Error covariance
        S_k = H_k @ P_k @ H_k.T + R_k
        #   2.3. Update Kalman filter gain
        K_k = P_k @ H_k.T @ np.linalg.inv(S_k)

        #   2.4. State update
        x_est[:, k] = x_pred[:, k] + K_k @ z_error[:, k]

        #   2.5. State coraviance
        P_k = (np.eye(STATE_SIZE) - K_k @ H_k) @ P_k
        
        print(f"state estimation : {x_est[:3, k]}")
        print(f"state truth : {x_true[:3, k]}")


import matplotlib.pyplot as plt
from state_display import display_3D_traj, display_2D_traj

trajectories = [
        {"label": "True Trajectory",      "data": x_true[:3]},
        {"label": "Estimated Method 1",   "data": x_est[:3]}
    ]

space_size = np.array([[-50, 50], [-50, 50], [-20, 20]])
#space_size = np.array([[-50, 0], [-50, 0], [20, 0]])
display_3D_traj(trajectories, space_size = space_size)
display_2D_traj(trajectories, space_size = space_size, XY = True, XZ = True, YZ = True)


plt.figure()
for k in range(STATE_SIZE):
    plt.plot(t, x_true[k] - x_est[k], label = {k})

plt.legend()
plt.show()
