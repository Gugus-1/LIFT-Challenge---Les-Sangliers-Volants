import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from state_display import display_3D_traj, display_2D_traj

# --- CONSTANTS ---
MASS = 1.5          # kg
GRAVITY = np.array([0, 0, -9.81])  # WORLD frame

# state layout: pos(3), vel(3), quat(4)= [x,y,z,w], acc_bias(3), gyro_bias(3), mag_bias(1), omega(3), baro(1)
IDX_POS     = slice(0,  3)
IDX_VEL     = slice(3,  6)
IDX_QUAT    = slice(6, 10)
IDX_ACC_B1  = slice(10,13)
IDX_GY_B1   = slice(13,16)
IDX_MAG_B1 = slice(16, 19)
IDX_ACC_B2  = slice(19, 22)
IDX_GY_B2   = slice(22,25)
IDX_MAG_B2= slice(25, 28)

IDX_OMEGA   = slice(28, 31)
IDX_BARO_B = slice(31, 32)
STATE_SIZE  = 32

# motor geometry unchanged
MOTOR_DIRS = np.array([[ 1,  1, 0],
                       [ 0,  0, 1],
                       [ 1, -1, 0],
                       [ 0,  0, 1],
                       [-1, -1, 0],
                       [ 0,  0, 1],
                       [-1,  1, 0],
                       [ 0,  0, 1]])
MOTOR_POS = np.array([
    [ 0.5,  1, 0.0],
    [ 0.5,  1, 0.0],
    [ 0.5, -1, 0.0],
    [ 0.5, -1, 0.0],
    [-0.5, -1, 0.0],
    [-0.5, -1, 0.0],
    [-0.5,  1, 0.0],
    [-0.5,  1, 0.0],
])

ANCHOR_POS = np.array([
    [ 0.0,   0.0,   20.0],
    [200.0,   0.0,   2.0],
    [200.0,  200.0,   20.0],
    [ 0.0,  200.0,   2.0],
    [100.0,   0.0,   2.0],
    [200.0,  100.0,   20.0],
    [100.0,  200.0,   2.0],
    [ 0.0,  100.0,   20.0],
])

TAG_POS = np.array([
    [1.5, 0, 0],
    [-1.5, 0, 0]
])

IMU_POS = [
    np.array([ 0.3, 0.02, 0.3]),   # IMU #1
    np.array([-0.3, -0.02, 0.3])    # IMU #2
]

IMU_DIR = [
    np.array([0.2, 0.2, 1]),
    np.array([-0.2, -0.2, 1])
]

Ixx, Iyy, Izz = 0.1, 0.12, 0.08
INERTIA = np.diag([Ixx, Iyy, Izz])
EPS = 1e-4

# world frame magnetic field vector
m_w = np.array([1/2, 1/2, 0])


# --- TRUE STATE PROPAGATION ---
def construct_true_state(x, u, dt):
    p        = x[IDX_POS]
    v        = x[IDX_VEL]
    q        = x[IDX_QUAT]
    acc_b1    = x[IDX_ACC_B1]
    gyro_b1   = x[IDX_GY_B1]
    mag_b1    = x[IDX_MAG_B1]
    acc_b2    = x[IDX_ACC_B2]    
    gyro_b2   = x[IDX_GY_B2]
    mag_b2    = x[IDX_MAG_B2]
    omega    = x[IDX_OMEGA]

    # position & velocity
    p_pred = p + v*dt
    F_body = MOTOR_DIRS.T @ u
    F_world= R.from_quat(q).apply(F_body)
    a_world= F_world/MASS + GRAVITY
    v_pred = v + a_world*dt

    # rotational dynamics
    forces  = MOTOR_DIRS * u[:,None]
    torques = np.cross(MOTOR_POS, forces)
    tau     = torques.sum(axis=0)
    omega_dot = np.linalg.inv(INERTIA) @ (tau - np.cross(omega, INERTIA@omega))
    omega_pred = omega + omega_dot*dt

    # 4) Quaternion (do the same scalar‐first ↔ scalar‐last swap)
    #  — build a proper SciPy rotation from your [w,x,y,z]
    r_old   = R.from_quat(q)
    dq_rot  = R.from_rotvec(omega_pred * dt)
    r_new   = r_old * dq_rot
    q_pred  = r_new.as_quat()        

    qn = np.linalg.norm(q_pred)
    q_pred = q_pred/qn if qn>1e-8 else np.array([1.,0.,0.,0.])

    # biases random‐walk (unchanged)
    #TODO: is this true ?
    acc1_b_pred  = acc_b1
    acc2_b_pred  = acc_b2
    gyro1_b_pred = gyro_b1
    gyro2_b_pred = gyro_b2
    mag_b1_preq = mag_b1
    mag_b2_preq = mag_b2
    x_new = np.zeros(STATE_SIZE)
    x_new[IDX_POS]    = p_pred
    x_new[IDX_VEL]    = v_pred
    x_new[IDX_QUAT]   = q_pred
    x_new[IDX_ACC_B1]  = acc1_b_pred
    x_new[IDX_ACC_B2]  = acc2_b_pred
    x_new[IDX_GY_B1]   = gyro1_b_pred
    x_new[IDX_GY_B2]   = gyro2_b_pred
    x_new[IDX_MAG_B1]   = mag_b1_preq
    x_new[IDX_MAG_B2]   = mag_b2_preq
    x_new[IDX_OMEGA]  = omega_pred
    return x_new

# --- FILTER PREDICTION ---
def predict_next_state(x, u, dt):
    p       = x[IDX_POS]  
    v       = x[IDX_VEL]
    q       = x[IDX_QUAT]
    acc_b1  = x[IDX_ACC_B1]
    gyro_b1 = x[IDX_GY_B1]
    acc_b2  = x[IDX_ACC_B2]
    gyro_b2 = x[IDX_GY_B2]
    mag_b1    = x[IDX_MAG_B1]
    mag_b2    = x[IDX_MAG_B2]
    meas_omega = x[IDX_OMEGA]

    # 1) translational
    p_pred = p + v*dt
    F_body = MOTOR_DIRS.T @ u
    Fw     = R.from_quat(q).apply(F_body)
    v_pred = v + (Fw/MASS + GRAVITY)*dt

    # 2) rotational
    forces = MOTOR_DIRS * u[:,None]
    tau    = np.cross(MOTOR_POS, forces).sum(axis=0)
    omega_dot = np.linalg.inv(INERTIA) @ (tau - np.cross(meas_omega, INERTIA @ meas_omega))
    omega_pred = meas_omega + omega_dot*dt

    # 3) Quaternion
    r_old   = R.from_quat(q)
    dq_rot  = R.from_rotvec(omega_pred * dt)
    q_pred   = (r_old * dq_rot).as_quat()  
    q_pred /= np.linalg.norm(q_pred)

     # 4) biases (random walk or fixed)
    x_pred = np.zeros(STATE_SIZE)
    x_pred[IDX_POS]   = p_pred
    x_pred[IDX_VEL]   = v_pred
    x_pred[IDX_QUAT]  = q_pred
    x_pred[IDX_ACC_B1] = acc_b1
    x_pred[IDX_ACC_B2] = acc_b2
    x_pred[IDX_GY_B1]  = gyro_b1
    x_pred[IDX_GY_B2]  = gyro_b2
    x_pred[IDX_OMEGA]= omega_pred
    x_pred[IDX_MAG_B1]   = mag_b1
    x_pred[IDX_MAG_B2]   = mag_b2
    return x_pred

# --- JACOBIAN OF f ---
def compute_F(x, u, dt):
    F = np.zeros((STATE_SIZE,STATE_SIZE))
    F[IDX_ACC_B1, IDX_ACC_B1] = np.eye(3)
    F[IDX_GY_B1, IDX_GY_B2] = np.eye(3)
    F[IDX_MAG_B1, IDX_MAG_B2] = np.eye(3)

    f0 = predict_next_state(x,u,dt)
    for i in range(STATE_SIZE):
        dx = np.zeros(STATE_SIZE); dx[i]=EPS
        f1 = predict_next_state(x+dx,u,dt)
        F[:,i] = (f1-f0)/EPS
    return F

# --- MEASUREMENT MODELS ---
def h_accel(x_mes, prev_mes, dt):
    # 1) compute CG world‐frame accel:
    v_dot = (x_mes[IDX_VEL] - prev_mes[IDX_VEL]) / dt
    a_cg_w = v_dot - GRAVITY

    # 2) get ω and ω̇ in world
    ω     = x_mes[IDX_OMEGA]
    # approximate ω̇ by FD of ω:
    ω_dot = (x_mes[IDX_OMEGA] - prev_mes[IDX_OMEGA]) / dt

    Rwb = R.from_quat(x_mes[IDX_QUAT]).as_matrix()
    Rbw = Rwb.T

    out = []
    for i, r_body in enumerate(IMU_POS):
        # world‐frame offset
        r_w = Rwb @ r_body

        # sensor world accel
        a_s_w = a_cg_w \
              + np.cross(ω_dot, r_w) \
              + np.cross(ω, np.cross(ω, r_w))

        # back to body‐frame
        a_s_b = Rbw @ a_s_w

        # subtract its bias
        ab_slice = IDX_ACC_B1 if i==0 else IDX_ACC_B2
        a_bias   = x_mes[ab_slice]
        out.append(a_s_b - a_bias)

    return np.hstack(out)

def h_gyro(x_mes):
    # measures ω + bias
    return np.hstack([x_mes[IDX_OMEGA] + x_mes[IDX_GY_B1], x_mes[IDX_OMEGA] + x_mes[IDX_GY_B2]])

def h_baro(x_mes):
    return np.array([x_mes[IDX_POS][2] + x_mes[IDX_BARO_B.start]])

def h_uwb(x_mes):
    p = x_mes[IDX_POS]
    Rwb = R.from_quat(x_mes[IDX_QUAT]).as_matrix()  # mind scalar‐last
    # body→world offsets:
    pf = p + Rwb @ TAG_POS[0]
    pr = p + Rwb @ TAG_POS[1]
    # ranges:
    r_front = np.linalg.norm(pf - ANCHOR_POS, axis=1)
    r_rear  = np.linalg.norm(pr - ANCHOR_POS, axis=1)

    return np.hstack([r_front, r_rear])

def h_mag(x_mes):
    q = x_mes[IDX_QUAT]
    rbw = R.from_quat(q).as_matrix().T @ m_w
    m1 = rbw + x_mes[IDX_MAG_B1]
    m2 = rbw + x_mes[IDX_MAG_B2]
    return np.hstack([m1, m2]) 

def compute_H_mag(x, eps=1e-6):
    H = np.zeros((6, STATE_SIZE))
    q0 = x[IDX_QUAT]
    for j in range(4):
        dq = np.zeros(4); dq[j] = eps
        # forward/back
        m_fw = R.from_quat(q0 + dq).as_matrix().T @ m_w
        m_bw = R.from_quat(q0 - dq).as_matrix().T @ m_w
        dmdq = (m_fw - m_bw) / (2*eps)

        # same derivative for both IMUs
        H[0:3, 6+j] = dmdq
        H[3:6, 6+j] = dmdq
    
    H[0:3, IDX_MAG_B1] = np.eye(3)
    H[3:6, IDX_MAG_B2] = np.eye(3)

    return H

def h_combined(x_pred, x_prev, dt):
    a_pred = h_accel(x_pred, x_prev, dt)
    g_pred = h_gyro(x_pred)
    b_pred = h_baro(x_pred)
    r_pred = h_uwb(x_pred)
    m_pred = h_mag(x_pred)
    return np.hstack([ a_pred, g_pred, b_pred, r_pred, m_pred])

def compute_H_acc(x_pred, x_prev, dt):
    Rbw = R.from_quat(x_pred[IDX_QUAT]).as_matrix().T
    dv  = (x_pred[IDX_VEL] - x_prev[IDX_VEL]) / dt

    H = np.zeros((3*2, STATE_SIZE))    # 2 IMUs → 6 rows

    for i in range(2):
        rows       = slice(3*i, 3*i+3)
        bias_slice = IDX_ACC_B1 if i==0 else IDX_ACC_B2

        # 1) ∂h/∂v
        H[rows, IDX_VEL] = Rbw / dt

        # 2) ∂h/∂q (finite diff)
        eps = 1e-6
        for j in range(4):
            dq       = np.zeros(4); dq[j] = eps
            rp       = R.from_quat(x_pred[IDX_QUAT]+dq).as_matrix().T @ dv
            rm       = Rbw @ dv
            H[rows, 6+j] = (rp - rm) / eps

        # 3) ∂h/∂bias_i
        H[rows, bias_slice] = -np.eye(3)

    return H

def compute_H_gyro():
    H = np.zeros((6, STATE_SIZE))
    for i in range(2):
        base = 3*i
        H[base:base+3, IDX_OMEGA] = np.eye(3)
        bias_slice = IDX_GY_B1 if i==0 else IDX_GY_B2
        H[base:base+3, bias_slice] = np.eye(3)
    return H


def compute_H_baro():
    H = np.zeros((1,STATE_SIZE))
    H[0,IDX_POS.start+2] = 1.0
    H[0, IDX_BARO_B.start]   = -1.0
    return H

def compute_H_uwb(x_pred):
    p = x_pred[IDX_POS]
    ranges = np.linalg.norm(p-ANCHOR_POS,axis=1)

    r = R.from_quat(x_pred[IDX_QUAT])
    Rwb = r.as_matrix()      # world ← body

    pf = p + Rwb @ TAG_POS[0]
    pr = p + Rwb @ TAG_POS[1]

    H = np.zeros((8,STATE_SIZE))
    # stack them so we can loop
    tag_world = [pf, pr]
    H = np.zeros((16, STATE_SIZE))

    for t, pw in enumerate(tag_world):
        # ranges from this tag to each of the 8 anchors
        diffs = pw[None,:] - ANCHOR_POS    # shape (8,3)
        ranges = np.linalg.norm(diffs, axis=1)  # (8,)

        for i in range(8):
            row = t*8 + i
            # ∂r/∂p = (pw - a_i)/range_i
            H[row, IDX_POS] = diffs[i] / ranges[i]

            # ∂r/∂q enters via ∂(Rwb·v_t)/∂q
            # you can finite-difference it:
            eps = 1e-6
            # create a block of size 4 for the quaternion derivative
            dq_block = np.zeros(4)
            for j in range(4):
                dq = np.zeros(4); dq[j] = eps
                # perturb scalar-first quaternion
                # build corresponding Rwb_pert
                r_p = R.from_quat(x_pred[IDX_QUAT]+ dq)
                pw_pert = p + r_p.as_matrix() @ (TAG_POS[0] if t==0 else TAG_POS[1])
                diff_pert = pw_pert - ANCHOR_POS[i]
                range_pert = np.linalg.norm(diff_pert)
                # finite-difference row entry
                dq_block[j] = (range_pert - ranges[i]) / eps
            H[row, 6:10] = dq_block
    return H

# --- COMMAND SCHEDULE (unchanged) ---
def command_generation(k):


    cmd1 = np.array([1.01, 3.68, 1, 3.68, -1, 3.68, -1, 3.68])
    #cmd1 = np.array([10, 3.68, -10, 3.68, -10, 3.68, 10, 3.68])
    #cmd1 = np.array([0, 3.65, 0, 3.65, 0, 3.65, 0, 3.65])
    cmd2 = np.array([10, 3, 10, 3, -10, 3, -10, 3])
    cmd3 = np.array([-10, 4, 10, 4, 10, 4, -10, 4])
    cmd4 = np.array([10, 4, -10, 4, -10, 4, -10, 4])
    cmd5 = np.array([-20, 3, -10, 3, 10, 3, -15, 3])
    cmd6 = np.array([-10, 3.5, 10, 3.5, -10, 3.5, -15, 3.5])

    if k<200:   return cmd1
    elif k < 210: return cmd1 + cmd2
    elif k<400:   return cmd2
    elif k < 410: return cmd2 + cmd2
    elif k<600:   return cmd3
    elif k < 610: return cmd3 + cmd4
    elif k<800:   return cmd4
    elif k < 810: return cmd4 + cmd5 + np.array([0, 0, 0, 0, 0, 0, 0, 0.001])
    elif k<800:   return cmd5
    elif k < 1010: return cmd5 + cmd6
    elif k<1000:   return cmd6

    



    #if k<100:   return np.array([ 0,2, 0,2, 0, 2, 0,2])
   
    return         np.zeros(8)


def main():
    T  = 700
    t  = np.linspace(0,10,T)
    dt = 0.001

    x_true = np.zeros((STATE_SIZE,T))
    x_est  = np.zeros_like(x_true)
    x_pred = np.zeros_like(x_true)

    x_true[IDX_QUAT,0] = [0.,0.,0.,1.]
    x_est [IDX_QUAT,0] = [0.,0.,0.,1.]

    # TODO:update -- measurement dims: 6(acc)+6(gyro)+6(mag)+1(baro)+2*8(anchors)=29
    z_meas  = np.zeros((35,T))
    z_pred  = np.zeros_like(z_meas)
    z_error = np.zeros_like(z_meas)

        

    # commands
    command = np.zeros((8,T))
    for k in range(T):
        command[:,k] = command_generation(k)

    # Propagate x_true
    for k in range(1, T):
        x_true[:, k] = construct_true_state(x_true[:, k-1], command[:, k-1], dt)

    # --- Process Noise (Q) ---

    process_noise_std = {
        # we assume our dynamics model is pretty good in pos/vel…
        "pos":     1e-3,    # ~1 mm of unmodeled position drift per step
        "vel":     1e-1,    # ~0.1 m/s of unmodeled accel per step
        # allow modest quaternion drift
        "quat":    1e-2,    # corresponds to ~0.6° of unmodeled rotation
        # bias random‐walks (typical IMU bias instability)
        "acc_b1":  1e-3,    # ~1 mg/sqrt(Hz)
        "gyro_b1": 1e-4,    # ~0.005°/s/sqrt(Hz)
        "mag_b1":  1e-4,    # small magnetometer bias drift
        "acc_b2":  1e-3,
        "gyro_b2": 1e-4,
        "mag_b2":  1e-4,
        # we let angular‐rate wander a bit
        "omega":   1e-1,    # ~0.1 rad/s unmodeled
        # baro bias random‐walk
        "baro_b":  1e-2,    # ~1 cm of drift per sample
    }

    Q_k = np.diag(np.concatenate([
        np.ones(3) * process_noise_std["pos"]**2,
        np.ones(3) * process_noise_std["vel"]**2,
        np.ones(4) * process_noise_std["quat"]**2,
        np.ones(3) * process_noise_std["acc_b1"]**2,
        np.ones(3) * process_noise_std["gyro_b1"]**2,
        np.ones(3) * process_noise_std["mag_b1"]**2,
        np.ones(3) * process_noise_std["acc_b2"]**2,
        np.ones(3) * process_noise_std["gyro_b2"]**2,
        np.ones(3) * process_noise_std["mag_b2"]**2,
        np.ones(3) * process_noise_std["omega"]**2,
        np.ones(1) * process_noise_std["baro_b"]**2
    ]))

    # --- Measurement Noise (R) ---
    
    # MEAS NOISE R

    meas_noise_std = {
        # accelerometer noise density ~200 µg/√Hz → σ ≈ 0.06 m/s² per 1 kHz sample
        "acc_imu1":   0.06,     
        "gyro_imu1":  np.deg2rad(0.3),  # ~0.3°/s noise
        "mag_imu1":   0.01,     # ~1% of Earth's field
        "acc_imu2":   0.06,
        "gyro_imu2":  np.deg2rad(0.3),
        "mag_imu2":   0.01,
        # barometer: good sensor ~0.1 m of altitude noise
        "baro":       0.1,      
        # UWB ranging: sub-decimeter accuracy
        "uwb":        0.05,     # 5 cm σ
    }

    R_k = np.diag(np.concatenate([
        np.ones(3) * meas_noise_std["acc_imu1"]**2,
        np.ones(3) * meas_noise_std["gyro_imu1"]**2,
        np.ones(3) *meas_noise_std["mag_imu1"]**2,
        np.ones(3) * meas_noise_std["acc_imu2"]**2,
        np.ones(3) * meas_noise_std["gyro_imu2"]**2,
        np.ones(3) *meas_noise_std["mag_imu2"]**2,
        [meas_noise_std["baro"]**2],
        np.ones(8) * meas_noise_std["uwb"]**2,
        np.ones(8) * meas_noise_std["uwb"]**2
    ]))

    # --- Initial Covariance (P) ---

    init_state_std = {
        "pos":       5.0,          # we might start off ±5 m uncertain
        "vel":       2.0,          # ±2 m/s
        "quat":      np.deg2rad(10),  # ±10°
        # assume we don’t know biases well initially
        "acc_b1":    0.1,          # ±0.1 m/s²
        "gyro_b1":   np.deg2rad(1),# ±1°/s
        "mag_b1":    0.05,         # ±5% of field
        "acc_b2":    0.1,
        "gyro_b2":   np.deg2rad(1),
        "mag_b2":    0.05,
        "omega":     1.0,          # ±1 rad/s
        "baro_b":    1.0,          # ±1 m of baro bias
    }

    P_k = np.diag(np.concatenate([
        np.ones(3) * init_state_std["pos"]**2,
        np.ones(3) * init_state_std["vel"]**2,
        np.ones(4) * init_state_std["quat"]**2,
        np.ones(3) * init_state_std["acc_b1"]**2,
        np.ones(3) * init_state_std["gyro_b1"]**2,
        np.ones(3) * init_state_std["mag_b1"]**2,
        np.ones(3) * init_state_std["acc_b2"]**2,
        np.ones(3) * init_state_std["gyro_b2"]**2,
        np.ones(3) * init_state_std["mag_b2"]**2,
        np.ones(3) * init_state_std["omega"]**2,
        np.ones(1) * init_state_std["baro_b"]**2
    ]))


    P_hist = np.zeros((STATE_SIZE, T))

    assert MASS == 1.5
    assert np.allclose(INERTIA, np.diag([Ixx,Iyy,Izz]))


    # KF loop
    for k in range(1,T):

        # predict
        x_pred[:,k] = predict_next_state(x_est[:,k-1], command[:,k-1], dt)
        #x_pred[:,k] = construct_true_state(x_est[:,k-1], command[:,k-1], dt)
        err = np.linalg.norm(x_pred[:3, k] - x_true[:3,k])
        err_v = np.linalg.norm(x_pred[IDX_VEL, k] - x_true[IDX_VEL,k])
        err_q = np.linalg.norm(x_pred[IDX_QUAT, k] - x_true[IDX_QUAT,k])
       #print(f"{k:3d} | Δp={err:.3e} Δv={err_v:.3e} Δq={err_q:.3e}")


        F_body = MOTOR_DIRS.T @ command[:,k-1]
        Fw_sim   = R.from_quat(x_true[IDX_QUAT, k]).apply(F_body)
        a_sim   = Fw_sim / MASS + GRAVITY

        Fw_pred  = R.from_quat(x_pred[IDX_QUAT, k]).apply(F_body)
        a_pred  = Fw_pred / MASS + GRAVITY
        #print("a_sim", a_sim, "a_pred", a_pred)

        
        F_k         = compute_F(x_est[:,k-1], command[:,k-1], dt)
        P_k         = F_k @ P_k @ F_k.T + Q_k

        # build H
        H_acc    = compute_H_acc(x_pred[:,k], x_est[:,k-1], dt)
        H_gyro   = compute_H_gyro()
        H_baro   = compute_H_baro()
        H_tag = compute_H_uwb(x_pred[:,k])
        H_mag = compute_H_mag(x_pred[:, k])

        dq_yaw = np.array([0,0,1e-6,0])   # [x,y,z,w] format → rotation about z
        # project H_uwb onto dq_yaw:
        sens = H_tag[:,6:10] @ dq_yaw
        #print("UWB yaw sensitivity norm:", np.linalg.norm(sens))


        H_k      = np.vstack([H_acc, H_gyro, H_baro, H_tag, H_mag])

        # simulate meas
        acc_meas = h_accel(x_true[:,k], x_true[:,k-1], dt) - np.hstack([x_pred[IDX_ACC_B1, k], x_pred[IDX_ACC_B2, k]]) + np.random.randn(6)*meas_noise_std["acc_imu1"]

        gyro_meas = h_gyro(x_true[:,k]) - np.hstack([x_pred[IDX_GY_B1, k], x_pred[IDX_GY_B2, k]]) + np.random.randn(6)*meas_noise_std["gyro_imu1"]

        baro_meas = h_baro(x_true[:,k]) - x_pred[IDX_BARO_B.start, k] + np.random.randn()*meas_noise_std["baro"]

        tag_meas = h_uwb(x_true[:, k]) + np.random.randn(16)*meas_noise_std["uwb"]

        mag_meas = h_mag(x_true[:, k]) - np.hstack([x_pred[IDX_MAG_B1, k], x_pred[IDX_MAG_B2, k]]) + np.random.randn(6)*meas_noise_std["mag_imu1"]

        z_meas[:,k] = np.hstack([acc_meas, gyro_meas, baro_meas, tag_meas, mag_meas])

        # update
        z_pred[:,k]  = h_combined(x_pred[:,k], x_est[:,k-1], dt)
        z_error[:,k] = z_meas[:,k] - z_pred[:,k]
        S_k          = H_k @ P_k @ H_k.T + R_k
        cond_S = np.linalg.cond(S_k)
        #print(f"Step {k}: cond(S_k) = {cond_S:.2e}")
        K_k          = P_k @ H_k.T @ np.linalg.inv(S_k)
        x_est[:,k]   = x_pred[:,k] + K_k @ z_error[:,k]
        P_k          = (np.eye(STATE_SIZE) - K_k @ H_k) @ P_k

        print(f"iter {k:3d} | est pos {x_est[:3,k]} | true pos {x_true[:3,k]}")
    
        r_est = R.from_quat(x_est[IDX_QUAT, k])
        r_true = R.from_quat(x_true[IDX_QUAT, k])

        print(f"iter {k:3d} | est angle {r_est.as_euler('xyz', degrees=True)} | true angle {r_true.as_euler('xyz', degrees=True)}")

        P_hist[:, k] = np.diag(P_k)

        J = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0]])  # this is a rough placeholder

        P_q = P_k[IDX_QUAT, IDX_QUAT]
        var_yaw = J[2] @ P_q @ J[2].T
        #print(f"Step {k}: yaw variance = {var_yaw:.3e}")

        #x_est[:, k] = x_pred[:, k]



    # ─── 1) LOG PURE‐UWB TAG POSITIONS ────────────────────────────────────────
    pf_hist = np.zeros((3, T))
    pr_hist = np.zeros((3, T))
    for k in range(T):
        qk   = x_true[IDX_QUAT, k]
        Rwb  = R.from_quat(qk).as_matrix()
        pk   = x_true[IDX_POS, k]
        pf_hist[:,k] = pk + Rwb @ TAG_POS[0]
        pr_hist[:,k] = pk + Rwb @ TAG_POS[1]

    # ─── 2) PURE GYRO INTEGRATION FOR ORIENTATION ─────────────────────────────
    q_gyro = np.zeros((4, T))
    q_gyro[:,0] = x_est[IDX_QUAT,0].copy()    # start from your known init
    for k in range(1, T):
        # take the first IMU’s 3‐axis gyro meas from z_meas
        ω_meas = z_meas[3:6, k]               # or average both IMUs if you like
        dq     = R.from_rotvec(ω_meas * dt)
        q_prev = R.from_quat(q_gyro[:,k-1])
        q_gyro[:,k] = (q_prev * dq).as_quat()

    # ─── 3) PURE ACCEL INTEGRATION FOR POSITION ───────────────────────────────
    vel_acc = np.zeros((3, T))
    pos_acc = np.zeros((3, T))
    for k in range(1, T):
        a_b = z_meas[0:3, k]                  # first IMU accel
        a_w = R.from_quat(q_gyro[:,k-1]).apply(a_b) + GRAVITY
        vel_acc[:,k] = vel_acc[:,k-1] + a_w * dt
        pos_acc[:,k] = pos_acc[:,k-1] + vel_acc[:,k] * dt

    # ─── 4) PURE MAG INTEGRATION FOR POSITION ───────────────────────────────
    # ─── MAGNETOMETER: RAW vs. TRUE BODY-FRAME FIELD ─────────────────────────
    # meas indices: acc(0:6), gyro(6:12), baro(12), uwb(13:29), mag(29:35)
    mag1_meas = z_meas[29:32, :]   # IMU1
    mag2_meas = z_meas[32:35, :]   # IMU2

    # compute the ’true’ body-frame magnetic field (same for both IMUs)
    m_true_b = np.zeros_like(mag1_meas)
    for k in range(T):
        Rbw = R.from_quat(x_true[IDX_QUAT, k]).as_matrix().T
        m_true_b[:, k] = Rbw @ m_w

    # plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    t = np.arange(T) * dt
    for i, ax in enumerate(axs):
        # choose which IMU
        meas = mag1_meas if i == 0 else mag2_meas
        ax.plot(t, meas.T,    linestyle='-',  alpha=0.6)
        ax.plot(t, m_true_b.T, linestyle='--', color='k', linewidth=1)
        ax.set_title(f'IMU {i+1} Magnetometer (raw vs. true)')
        ax.set_ylabel('Mag field (arb. units)')
        ax.grid(True)
        ax.legend(['Mx','My','Mz','true M'])
    axs[1].set_xlabel('Time (s)')

    # ─── PLOTTING ───────────────────────────────────────────────────────────
    # plot the raw UWB‐tag world positions
    plt.figure(figsize=(6,6))
    plt.scatter(ANCHOR_POS[:,0], ANCHOR_POS[:,1], marker='x', color='k', label='Anchors')
    plt.plot(pf_hist[0], pf_hist[1],  label='UWB front tag (raw)',   alpha=0.7)
    plt.plot(pr_hist[0], pr_hist[1],  label='UWB rear tag (raw)',    alpha=0.7)
    plt.plot(x_true[0], x_true[1],    'k--', label='True pos')
    plt.legend(); plt.xlabel('X'); plt.ylabel('Y'); plt.title('Raw UWB tag positions')

    # plot orientation comparison
    e_true = np.array([R.from_quat(x_true[IDX_QUAT, k]).as_euler('xyz', degrees=True)
                       for k in range(T)]).T
    e_gyro = np.array([R.from_quat(q_gyro[:,k]).as_euler('xyz', degrees=True)
                       for k in range(T)]).T

    plt.figure(figsize=(10,6))
    for i, lab in enumerate(['Roll','Pitch','Yaw']):
        plt.plot(e_true[i], label=f'{lab} true')
        plt.plot(e_gyro[i], '--', label=f'{lab} gyro‐only')
    plt.legend(); plt.title('Euler angles: true vs. gyro‐integrated'); plt.xlabel('step')

    # plot pure accel‐dead‐reckoning
    plt.figure(figsize=(6,6))
    plt.plot(pos_acc[0], pos_acc[1], label='IMU dead‐reckoning')
    plt.plot(x_true[0], x_true[1], 'k--', label='True')
    plt.legend(); plt.title('Pure IMU accel‐DR trajectory'); plt.xlabel('X'); plt.ylabel('Y')

    #plt.show()


    # Preallocate arrays to store Euler angles
    euler_true = np.zeros((3, T))
    euler_est  = np.zeros((3, T))

    # Convert quaternions to Euler angles (xyz convention, in degrees)
    for k in range(T):
        r_true = R.from_quat(x_true[IDX_QUAT, k])
        euler_true[:, k] = r_true.as_euler('xyz', degrees=True)

        r_est = R.from_quat(x_est[IDX_QUAT, k])
        euler_est[:, k] = r_est.as_euler('xyz', degrees=True)

    # Plot angles
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['Roll (°)', 'Pitch (°)', 'Yaw (°)']
    for i in range(3):
        axs[i].plot(np.arange(T), euler_true[i], label='True', color='k')
        axs[i].plot(np.arange(T), euler_est[i], label='Estimated', linestyle='--')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
    axs[2].set_xlabel('iter')
    axs[0].legend()
    plt.suptitle("Euler Angle Comparison (XYZ order)")
    plt.tight_layout()


    # VISUALIZE
    #                           6           6           3           6       6
    # z_meas[:,k] = np.hstack([acc_meas, gyro_meas, baro_meas, tag_meas, mag_meas])
    trajs = [
        {"label":"True Trajectory",   "data": x_true[:3]},
        {"label":"EKF", "data": x_est[:3]},
        {"label": "UWB front - EKF", "data": z_meas[15:18, :]},
        {"label": "UWB back - EKF", "data": z_meas[18:21, :]},
        {"label": "UWB front - raw", "data": pf_hist},
        {"label": "UWB back - raw", "data": pr_hist},
        
    ]
    #{"label": "Baro-only Trajectory",  "data": np.vstack([x_true[0, :],_true[1, :], z_meas[12, :]])},
    space_size = np.array([[-50,200],[-50,200],[-5,30]])
    #space_size = None
    display_2D_traj(trajs, space_size=space_size, XY=True, XZ=True, YZ=True)

    # plot error
    plt.figure()
    for i in range(STATE_SIZE):
        plt.plot(t, x_true[i]-x_est[i], label=str(i))
    plt.legend()

    display_3D_traj(trajs, space_size=space_size)


    # Extract data
    acc1_bias = x_est[IDX_ACC_B1, :]
    gyro1_bias = x_est[IDX_GY_B1, :]
    acc2_bias = x_est[IDX_ACC_B2, :]
    gyro2_bias = x_est[IDX_GY_B2, :]

    baro_bias = x_est[IDX_BARO_B, :]

    acc1_var = P_hist[IDX_ACC_B1, :]
    gyro1_var = P_hist[IDX_GY_B1, :]
    acc2_var = P_hist[IDX_ACC_B2, :]
    gyro2_var = P_hist[IDX_GY_B2, :]

    baro_var = P_hist[IDX_BARO_B, :]

    # Plotting helper
    def plot_bias_and_variance(bias, var, title_prefix, ylabel):
        axes = ['x', 'y', 'z']
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        for i in range(3):
            axs[0].plot(t, bias[i], label=f'{axes[i]}')
            axs[1].plot(t, var[i], label=f'{axes[i]}')

        axs[0].set_title(f'{title_prefix} Bias')
        axs[0].set_ylabel(ylabel)
        axs[0].grid(True)
        axs[0].legend()

        axs[1].set_title(f'{title_prefix} Variance')
        axs[1].set_ylabel(f'{ylabel}²')
        axs[1].set_xlabel('Time (s)')
        axs[1].grid(True)
        axs[1].legend()
        
        plt.tight_layout()

    def plot_baro_bias_and_variance(bias, var, title_prefix, ylabel):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        axs[0].plot(t, bias[0])
        axs[1].plot(t, var[0])

        axs[0].set_title(f'{title_prefix} Bias')
        axs[0].set_ylabel(ylabel)
        axs[0].grid(True)

        axs[1].set_title(f'{title_prefix} Variance')
        axs[1].set_ylabel(f'{ylabel}²')
        axs[1].set_xlabel('Time (s)')
        axs[1].grid(True)
        
        plt.tight_layout()

    # Plot all
    """ plot_bias_and_variance(acc1_bias, acc1_var, 'Accelerometer IMU 1', 'Bias (m/s²)')
    plot_bias_and_variance(gyro1_bias, gyro1_var, 'Gyroscope IMU 1', 'Bias (rad/s)')
    plot_bias_and_variance(acc2_bias, acc2_var, 'Accelerometer IMU 2', 'Bias (m/s²)')
    plot_bias_and_variance(gyro2_bias, gyro2_var, 'Gyroscope IMU 2', 'Bias (rad/s)')
    plot_baro_bias_and_variance(baro_bias, baro_var, 'Barometer', 'Bias (m)') """
    plt.show()











# --- MAIN SIM+KF LOOP ---
if __name__=="__main__":
    main()



