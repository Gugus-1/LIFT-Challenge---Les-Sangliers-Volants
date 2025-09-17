import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection


def display_3D_traj(trajectories, space_size = None):

    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    for traj in trajectories:
        x, y, z = traj["data"]
        ax3d.plot(x, y, z, label=traj["label"])
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.legend()
    ax3d.set_title("3D Trajectories")
    if space_size is not None:
        ax3d.set_xlim([space_size[0, 0], space_size[0, 1]])
        ax3d.set_ylim([space_size[1, 0], space_size[1, 1]])
        ax3d.set_zlim([space_size[2, 0], space_size[2, 1]])


def display_2D_traj(trajectories, XY = True, XZ = True, YZ = True, space_size = None):

    if XY is True:
        fig_xy = plt.figure()
        ax_xy = fig_xy.add_subplot(111)
        for traj in trajectories:
            x, y, _ = traj["data"]
            ax_xy.plot(x, y, label=traj["label"])
        ax_xy.set_xlabel('X')
        ax_xy.set_ylabel('Y')
        ax_xy.legend()
        ax_xy.set_title("XY Projection")
        if space_size is not None:
            ax_xy.set_xlim([space_size[0, 0], space_size[0, 1]])
            ax_xy.set_ylim([space_size[1, 0], space_size[1, 1]])
            


    if XZ is True:
        # XZ plane
        fig_xz = plt.figure()
        ax_xz = fig_xz.add_subplot(111)
        for traj in trajectories:
            x, _, z = traj["data"]
            ax_xz.plot(x, z, label=traj["label"])
        ax_xz.set_xlabel('X')
        ax_xz.set_ylabel('Z')
        ax_xz.legend()
        ax_xz.set_title("XZ Projection")
        if space_size is not None:
            ax_xz.set_xlim([space_size[0, 0], space_size[0, 1]])
            ax_xz.set_ylim([space_size[2, 0], space_size[2, 1]])

    # YZ plane
    if YZ is True:
        fig_yz = plt.figure()
        ax_yz = fig_yz.add_subplot(111)
        for traj in trajectories:
            _, y, z = traj["data"]
            ax_yz.plot(y, z, label=traj["label"])
        ax_yz.set_xlabel('Y')
        ax_yz.set_ylabel('Z')
        ax_yz.legend()
        ax_yz.set_title("YZ Projection")
        if space_size is not None:
            ax_yz.set_xlim([space_size[1, 0], space_size[1, 1]])
            ax_yz.set_ylim([space_size[2, 0], space_size[2, 1]])



if __name__ == "__main__":

    # ------------------------------------
    # 1) Generate or load your time series
    # ------------------------------------
    t = np.linspace(0, 10, 200)               # time vector
    x_true = np.sin(t)                        # true X
    y_true = np.cos(t)                        # true Y
    z_true = t                                # true Z

    # Example estimated trajectories (with noise)
    x_est1 = x_true + 0.1 * np.random.randn(len(t))
    y_est1 = y_true + 0.1 * np.random.randn(len(t))
    z_est1 = z_true + 0.1 * np.random.randn(len(t))

    x_est2 = x_true + 0.2 * np.random.randn(len(t))
    y_est2 = y_true + 0.2 * np.random.randn(len(t))
    z_est2 = z_true + 0.2 * np.random.randn(len(t))

    trajectories = [
        {"label": "True Trajectory",      "data": (x_true, y_true, z_true)},
        {"label": "Estimated Method 1",   "data": (x_est1, y_est1, z_est1)},
        {"label": "Estimated Method 2",   "data": (x_est2, y_est2, z_est2)},
        {"label": "Estimated Method 3",   "data": (x_est2 + x_est1, y_est2 + y_est1, z_est2 + z_est1)},
    ]

    display_3D_traj(trajectories)
    display_2D_traj(trajectories, XY = True, XZ = True, YZ = True)

    plt.show()