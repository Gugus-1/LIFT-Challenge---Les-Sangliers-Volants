import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from state_display import display_3D_traj, display_2D_traj

# --- CONSTANTS ---
MASS = 1.5          # kg
GRAVITY = np.array([0, 0, -9.81])  # WORLD frame

# state layout: pos(3), vel(3), quat(4)= [x,y,z,w], acc_bias(3), gyro_bias(3), omega(3)
IDX_POS     = slice(0,  3)
IDX_VEL     = slice(3,  6)
IDX_QUAT    = slice(6, 10)
IDX_ACC_B   = slice(10,13)
IDX_GY_B    = slice(13,16)
IDX_OMEGA   = slice(16,19)
STATE_SIZE  = 19

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
    [ 0.5,  0.5, 0.0],
    [ 0.5,  0.5, 0.0],
    [ 0.5, -0.5, 0.0],
    [ 0.5, -0.5, 0.0],
    [-0.5, -0.5, 0.0],
    [-0.5, -0.5, 0.0],
    [-0.5,  0.5, 0.0],
    [-0.5,  0.5, 0.0],
])
ANCHOR_POS = np.array([
    [ 0.0,   0.0,   2.0],
    [200.0,   0.0,   2.0],
    [200.0,  200.0,   2.0],
    [ 0.0,  200.0,   2.0],
    [100.0,   0.0,   2.0],
    [200.0,  100.0,   2.0],
    [100.0,  200.0,   2.0],
    [ 0.0,  100.0,   2.0],
])

TAG_POS = np.array([
    [1.5, 0, 0],
    [-1.5, 0, 0]
])

Ixx, Iyy, Izz = 0.1, 0.12, 0.08
INERTIA = np.diag([Ixx, Iyy, Izz])
EPS = 1e-4

# --- TRUE STATE PROPAGATION ---
def construct_true_state(x, u, dt):
    p        = x[IDX_POS]
    v        = x[IDX_VEL]
    q        = x[IDX_QUAT]
    acc_b    = x[IDX_ACC_B]
    gyro_b   = x[IDX_GY_B]
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
    acc_b_pred  = acc_b
    gyro_b_pred = gyro_b

    x_new = np.zeros(STATE_SIZE)
    x_new[IDX_POS]    = p_pred
    x_new[IDX_VEL]    = v_pred
    x_new[IDX_QUAT]   = q_pred
    x_new[IDX_ACC_B]  = acc_b_pred
    x_new[IDX_GY_B]   = gyro_b_pred
    x_new[IDX_OMEGA]  = omega_pred
    return x_new

# --- FILTER PREDICTION ---
def predict_next_state(x, u, dt):
    p      = x[IDX_POS];    v = x[IDX_VEL]
    q      = x[IDX_QUAT]
    acc_b  = x[IDX_ACC_B];  gyro_b = x[IDX_GY_B]
    meas_omega = x[IDX_OMEGA]

    # pos/vel
    p_pred = p + v*dt
    F_body = MOTOR_DIRS.T @ u
    Fw     = R.from_quat(q).apply(F_body)
    v_pred = v + (Fw/MASS + GRAVITY)*dt

    # torque & ω̇
    forces = MOTOR_DIRS * u[:,None]
    tau    = np.cross(MOTOR_POS, forces).sum(axis=0)
    omega  = meas_omega - gyro_b
    omega_dot = np.linalg.inv(INERTIA) @ (tau - np.cross(omega, INERTIA@omega))
    omega_pred = omega + omega_dot*dt

    # 4) Quaternion (do the same scalar‐first ↔ scalar‐last swap)
    #  — build a proper SciPy rotation from your [w,x,y,z]
    r_old   = R.from_quat(q)
    dq_rot  = R.from_rotvec(omega_pred * dt)
    r_new   = r_old * dq_rot
    q_pred      = r_new.as_quat()  
    qn = np.linalg.norm(q_pred)
    q_pred = q_pred/qn if qn>1e-8 else np.array([0.,0.,0.,1.])

    # biases fixed
    x_pred = np.zeros(STATE_SIZE)
    x_pred[IDX_POS]   = p_pred
    x_pred[IDX_VEL]   = v_pred
    x_pred[IDX_QUAT]  = q_pred
    x_pred[IDX_ACC_B] = acc_b
    x_pred[IDX_GY_B]  = gyro_b
    x_pred[IDX_OMEGA]= omega_pred
    return x_pred

# --- JACOBIAN OF f ---
def compute_F(x, u, dt):
    F = np.zeros((STATE_SIZE,STATE_SIZE))
    f0 = predict_next_state(x,u,dt)
    for i in range(STATE_SIZE):
        dx = np.zeros(STATE_SIZE); dx[i]=EPS
        f1 = predict_next_state(x+dx,u,dt)
        F[:,i] = (f1-f0)/EPS
    return F

# --- MEASUREMENT MODELS ---
def h_accel(x_mes, prev_mes, dt):
    # computes body‐frame accel minus bias
    dv = (x_mes[IDX_VEL]-prev_mes[IDX_VEL])/dt - GRAVITY
    Rbw= R.from_quat(x_mes[IDX_QUAT]).as_matrix().T
    return Rbw @ dv - x_mes[IDX_ACC_B]

def h_gyro(x_mes):
    # measures ω + bias
    return x_mes[IDX_OMEGA] + x_mes[IDX_GY_B]

def h_baro(x_mes):
    return np.array([ x_mes[IDX_POS][2] ])

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

def h_combined(x_pred, x_prev, dt):
    a_pred = h_accel(x_pred, x_prev, dt)
    g_pred = h_gyro(x_pred)
    b_pred = h_baro(x_pred)
    r_pred = h_uwb(x_pred)
    return np.hstack([ a_pred, g_pred, b_pred, r_pred ])

# --- H JACKS ---
def compute_H_acc(x_pred, x_prev, dt):
    H = np.zeros((3,STATE_SIZE))
    # ∂h/∂v
    Rbw = R.from_quat(x_pred[IDX_QUAT]).as_matrix().T
    H[:,IDX_VEL] = Rbw/dt
    # ∂h/∂q via FD
    dv = (x_pred[IDX_VEL]-x_prev[IDX_VEL])/dt
    eps=1e-6
    for i in range(4):
        dq = np.zeros(4); dq[i]=eps
        rp = R.from_quat(x_pred[IDX_QUAT]+dq).as_matrix().T@dv
        rm = Rbw@dv
        H[:,6+i] = (rp-rm)/eps
    # ∂h/∂bias
    H[:,IDX_ACC_B] = -np.eye(3)
    return H

def compute_H_gyro():
    H = np.zeros((3,STATE_SIZE))
    H[:,IDX_OMEGA] = np.eye(3)
    H[:,IDX_GY_B]  = np.eye(3)
    return H

def compute_H_baro():
    H = np.zeros((1,STATE_SIZE))
    H[0,IDX_POS.start+2] = 1.0
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
    return np.array([10, 3.68, -10, 3.68, 0, 3.68, 0, 3.68])
    if k<100:   return np.array([10, 4, -10, 4, -10, 4, 10, 4])
    elif k<200:   return np.array([10, 3, 10, 3, -10, 3, -10, 3])
    elif k<300:   return np.array([-10, 4, 10, 4, 10, 4, -10, 4])
    elif k<400:   return np.array([-10, 4, -10, 4, 10, 4, 10, 4])
    elif k<500:   return np.array([10, 10, -5, 10, -10, 10, 15, 10])
    elif k<600:   return np.array([-10, 1, -10, 0, -8, 0, -10, 1])

    



    #if k<100:   return np.array([ 0,2, 0,2, 0, 2, 0,2])
   
    return         np.zeros(8)

# --- MAIN SIM+KF LOOP ---
if __name__=="__main__":
    T  = 700
    t  = np.linspace(0,10,T)
    dt = 0.01

    x_true = np.zeros((STATE_SIZE,T))
    x_est  = np.zeros_like(x_true)
    x_pred = np.zeros_like(x_true)

    # measurement dims: 3(acc)+3(gyro)+1(baro)+2*8(anchors)=23
    z_meas  = np.zeros((23,T))
    z_pred  = np.zeros_like(z_meas)
    z_error = np.zeros_like(z_meas)

    # init quats
    for k in range(T):
        x_true[IDX_QUAT,k] = [0.,0.,0.,1.]
        x_est [IDX_QUAT,k] = [0.,0.,0.,1.]

    # commands
    command = np.zeros((8,T))
    for k in range(T):
        command[:,k] = command_generation(k)

    # PROCESS NOISE Q
    q_pos  = 1e-6; q_vel  = 1e-4; q_quat=1e-4
    q_ba   = 1e-8; q_bw   = 1e-6
    Q_k = np.diag(np.concatenate([
        np.ones(3)*q_pos,
        np.ones(3)*q_vel,
        np.ones(4)*q_quat,
        np.ones(3)*q_ba,    # accel bias
        np.ones(3)*q_bw,    # gyro bias
        np.ones(3)          # ω process noise
    ]))

    # MEAS NOISE R
    sig_acc    = 40e-6*9.81
    sig_gyro   = np.deg2rad(0.1)
    sig_baro   = 0.3
    sig_anchor = 0.1
    R_k = np.diag(np.concatenate([
        np.ones(3)*sig_acc**2,
        np.ones(3)*sig_gyro**2,
        [sig_baro**2],
        np.ones(8)*sig_anchor**2,
        np.ones(8)*sig_anchor**2
    ]))

    # STATE COVARIANCE P
    P_k = np.diag(np.concatenate([
        np.ones(3)*1.0,             # pos ±1m
        np.ones(3)*1.0,             # vel ±1m/s
        np.ones(4)*np.deg2rad(5.0)**2,  # att ±5°
        np.ones(3)*(0.1**2),        # accel bias
        np.ones(3)*(0.01**2),       # gyro bias
        np.ones(3)                  # ω
    ]))

    # KF loop
    for k in range(1,T):
        x_true[:,k] = construct_true_state(x_true[:,k-1], command[:,k-1], dt)

        F_body = MOTOR_DIRS.T @ command[:,k-1]

        # predict
        x_pred[:,k] = predict_next_state(x_est[:,k-1], command[:,k-1], dt)
        F_k         = compute_F(x_est[:,k-1], command[:,k-1], dt)
        P_k         = F_k @ P_k @ F_k.T + Q_k

        # build H
        H_acc    = compute_H_acc(x_pred[:,k], x_est[:,k-1], dt)
        H_gyro   = compute_H_gyro()
        H_baro   = compute_H_baro()
        h_tag = compute_H_uwb(x_pred[:,k])
        H_k      = np.vstack([H_acc, H_gyro, H_baro, h_tag])

        # simulate meas
        acc_meas = h_accel(x_true[:,k], x_true[:,k-1], dt) + np.random.randn(3)*sig_acc
        gyro_meas= h_gyro(x_true[:,k]) + np.random.randn(3)*sig_gyro
        baro_meas= x_true[IDX_POS, k][2] + np.random.randn()*sig_baro

        tag_meas = h_uwb(x_true[:, k]) + np.random.randn(16)*sig_anchor

        z_meas[:,k] = np.hstack([acc_meas, gyro_meas, baro_meas, tag_meas])

        # update
        z_pred[:,k]  = h_combined(x_pred[:,k], x_est[:,k-1], dt)
        z_error[:,k] = z_meas[:,k] - z_pred[:,k]
        S_k          = H_k @ P_k @ H_k.T + R_k
        K_k          = P_k @ H_k.T @ np.linalg.inv(S_k)
        x_est[:,k]   = x_pred[:,k] + K_k @ z_error[:,k]
        P_k          = (np.eye(STATE_SIZE) - K_k @ H_k) @ P_k

        print(f"iter {k:3d} | est pos {x_est[:3,k]} | true pos {x_true[:3,k]}")
    
        r_est = R.from_quat(x_est[IDX_QUAT, k])
        r_true = R.from_quat(x_true[IDX_QUAT, k])

        print(f"iter {k:3d} | est angle {r_est.as_euler('xyz', degrees=True)} | true angle {r_true.as_euler('xyz', degrees=True)}")



    # Preallocate arrays to store Euler angles
    euler_true = np.zeros((3, T))
    euler_est  = np.zeros((3, T))

    # Convert quaternions to Euler angles (xyz convention, in degrees)
    for k in range(T):
        qw, qx, qy, qz = x_true[IDX_QUAT, k]
        r_true = R.from_quat([qx, qy, qz, qw])
        euler_true[:, k] = r_true.as_euler('xyz', degrees=True)

        qw, qx, qy, qz = x_est[IDX_QUAT, k]
        r_est = R.from_quat([qx, qy, qz, qw])
        euler_est[:, k] = r_est.as_euler('xyz', degrees=True)

    # Plot angles
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['Roll (°)', 'Pitch (°)', 'Yaw (°)']
    for i in range(3):
        axs[i].plot(t, euler_true[i], label='True', color='k')
        axs[i].plot(t, euler_est[i], label='Estimated', linestyle='--')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
    axs[2].set_xlabel('Time (s)')
    axs[0].legend()
    plt.suptitle("Euler Angle Comparison (XYZ order)")
    plt.tight_layout()


    # VISUALIZE
    trajs = [
        {"label":"True Trajectory",   "data": x_true[:3]},
        {"label":"Estimated (1 IMU)", "data": x_est[:3]}
    ]
    space_size = np.array([[-50,200],[-50,200],[-5,30]])
    #space_size = None
    display_2D_traj(trajs, space_size=space_size, XY=True, XZ=True, YZ=True)

    # plot error
    plt.figure()
    for i in range(STATE_SIZE):
        plt.plot(t, x_true[i]-x_est[i], label=str(i))
    plt.legend()

    display_3D_traj(trajs, space_size=space_size)
    plt.show()
