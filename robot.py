import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

"""

Model for a 2-dof plannar robot. This follows the notation and
equations set out in the online course "Modern Robotics" by Kevin
Lynch and Frank Park.

https://www.youtube.com/watch?v=jVu-Hijns70&list=PLggLP4f-rq02vX0OQQ5vrCxbJrzamYDfx

"""

g = 9.81  # gravity
L1 = 1.  # length of link 1
L2 = 1.  # length of link 2
m1 = 1.5  # mass of link 1 (concentrated at elbow joint)
m2 = 1  # mass of link 2 (concentrated at end-effector)
Fd1 = 1  # friction damping for joint 1
Fd2 = 1  # friction damping for joint 2
theta1_lim = np.deg2rad([-135., 135.])  # position limits for joint 1
theta2_lim = np.deg2rad([-135., 135.])  # position limits for joint 2
robot_joint_radius = 0.1  # radii for the robot joints

def randtheta():
    rtheta1 = np.random.uniform(theta1_lim[0], theta1_lim[1])
    rtheta2 = np.random.uniform(theta2_lim[0], theta2_lim[1])
    return rtheta1, rtheta2

def x1(theta1, theta2):
    """position of elbow joint in x-axis"""
    return L1*cos(theta1)

def y1(theta1, theta2):
    """position of end-effector in y-axis"""
    return L1*sin(theta1)

def dx1(theta1, theta2, dtheta1, dtheta2):
    """velocity of elbow joint in x-axis"""
    return -L1*sin(theta1)*dtheta1

def dy1(theta1, theta2, dtheta1, dtheta2):
    """velocity of elbow joint in y-axis"""
    return L1*cos(theta1)*dtheta1

def x2(theta1, theta2):
    """position of end-effector in x-axis"""
    return L1*cos(theta1) + L2*cos(theta1+theta2)

def y2(theta1, theta2):
    """position of end-effector in y-axis"""
    return L1*sin(theta1) + L2*sin(theta1+theta2)

def dx2(theta1, theta2, dtheta1, dtheta2):
    """velocity of end-effector in x-axis"""
    return (-L1*sin(theta1) - L2*sin(theta1+theta2))*dtheta1 + (-L2*sin(theta1+theta2))*dtheta2

def dy2(theta1, theta2, dtheta1, dtheta2):
    """velocity of end-effector in y-axis"""
    return (L1*cos(theta1) + L2*cos(theta1+theta2))*dtheta1 + (L2*cos(theta1+theta2))*dtheta2

def J(theta1, theta2):
    """Jacobian matrix for end-effector"""
    return np.array([
        [-L1*sin(theta1) - L2*sin(theta1+theta2), -L2*sin(theta1+theta2)],
        [L1*cos(theta1) + L2*cos(theta1+theta2), L2*cos(theta1+theta2)],
    ])

def K1(theta1, theta2, dtheta1, dtheta2):
    """Kinetic energy for link 1"""
    return 0.5*m1*(dx1(theta1, theta2, dtheta1, dtheta2)**2 + dy1(theta1, theta2, dtheta1, dtheta2)**2)

def K2(theta1, theta2, dtheta1, dtheta2):
    """Kinetic energy for link 2"""
    return 0.5*m2*(dx2(theta1, theta2, dtheta1, dtheta2)**2 + dy2(theta1, theta2, dtheta1, dtheta2)**2)

def P1(theta1, theta2):
    """Potential energy for link 1"""
    return m1*g*y1(theta1, theta2)

def P2(theta1, theta2):
    """Potential energy for link 2"""
    return m2*g*y2(theta1, theta2)

def Lag(theta1, theta2, dtheta1, dtheta2):
    """Lagrangian function"""
    return (K1(theta1, theta2, dtheta1, dtheta2) - P1(theta1, theta2, dtheta1, dtheta2)) + (K2(theta1, theta2, dtheta1, dtheta2) - P2(theta1, theta2, dtheta1, dtheta2))

def tau1(theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2):
    """Compute required torque in joint 1 to achieve given joint position/velocity/acceleration (inverse dynamics)"""
    term1 = (m1*L1**2 + m2*(L1**2+2*L1*L2*cos(theta2)+L2**2))*ddtheta1 + m2*(L1*L2*cos(theta2)+L2**2)*ddtheta2
    term2 = -m2*L1*L2*sin(theta2)*(2*dtheta1*dtheta2+dtheta2**2)
    term3 = (m1+m2)*L1*g*cos(theta1) + m2*g*L2*cos(theta1 + theta2)
    return term1 + term2 + term3

def tau2(theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2):
    """Compute required torque in joint 2 to achieve given joint position/velocity/acceleration (inverse dynamics)"""
    term1 = m2*(L1*L2*cos(theta2)+L2**2)*ddtheta1 + m2*L2**2*ddtheta2
    term2 = m2*L1*L2*dtheta1**2*sin(theta2)
    term3 = m2*g*L2*cos(theta1+theta2)
    return term1 + term2 + term3

def M(theta1, theta2):
    """Inertia matrix"""
    return np.array([
        [m1*L1**2 + m2*(L1**2+2*L1*L2*cos(theta2)+L2**2), m2*(L1*L2*cos(theta2)+L2**2)],
        [                   m2*(L1*L2*cos(theta2)+L2**2),                     m2*L2**2],
    ])

def c(theta1, theta2, dtheta1, dtheta2):
    """Coriolis vector"""
    return np.array([-m2*L1*L2*sin(theta2)*(2*dtheta1*dtheta2+dtheta2**2), m2*L1*L2*dtheta1**2*sin(theta2)])

def gr(theta1, theta2):
    """Gravity term"""
    return np.array([(m1+m2)*L1*g*cos(theta1) + m2*g*L2*cos(theta1 + theta2), m2*g*L2*cos(theta1+theta2)])

def Fr(dtheta1, dtheta2):
    """Friction model"""
    return np.array([Fd1*dtheta1, Fd2*dtheta2])

def ID(theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2):
    """Inverse dynamics"""
    theta = np.array([theta1, theta2])
    dtheta = np.array([dtheta1, dtheta2])
    ddtheta = np.array([ddtheta1, ddtheta2])
    return M(theta1, theta2)@ddtheta + c(theta1, theta2, dtheta1, dtheta2) + gr(theta1, theta2)

def FD(theta1, theta2, dtheta1, dtheta2, tau1, tau2, apply_friction=True, apply_noise=False, eps=1e-3):
    """Forward dynamics"""
    theta = np.array([theta1, theta2])
    dtheta = np.array([dtheta1, dtheta2])
    tau = np.array([tau1, tau2])
    Minv = np.linalg.inv(M(theta1, theta2))
    N = c(theta1, theta2, dtheta1, dtheta2) + gr(theta1, theta2)
    if apply_friction:
        N += Fr(dtheta1, dtheta2)
    ddtheta = Minv @ (tau - N)
    ddtheta1 = ddtheta[0]
    ddtheta2 = ddtheta[1]

    if apply_noise:
        ddtheta1 += np.random.uniform(-eps, eps)
        ddtheta2 += np.random.uniform(-eps, eps)

    return ddtheta1, ddtheta2

def plot_robot(ax, theta1, theta2, set_axis_lims=True, add_coord_axis=True, grid=True):
    """Plot the robot"""

    X = [0, x1(theta1, theta2), x2(theta1, theta2)]
    Y = [0, y1(theta1, theta2), y2(theta1, theta2)]
    robot_plt, = ax.plot(X, Y, '-k', zorder=3, linewidth=4)
    circs = [Circle([x,y], radius=robot_joint_radius, color='blue', zorder=4) for x,y in zip(X, Y)]
    for c in circs:
        ax.add_patch(c)

    if add_coord_axis:
        zorder = 2
        ax.arrow(0, 0, 0, 0.5*L1, width=0.02, zorder=zorder, facecolor='green', edgecolor='green')
        ax.arrow(0, 0, 0.5*L1, 0, width=0.02, zorder=zorder, facecolor='red', edgecolor='red')

    if grid:
        ax.grid(zorder=1)

    if set_axis_lims:
        factor = 1.05
        d = (L1+L2)*factor
        ax.set_xlim(-d, d)
        ax.set_ylim(-d, d)
        ax.set_aspect('equal')

    return [robot_plt] + circs

def plot_trajectory(t, Theta1, Theta2, dTheta1=None, dTheta2=None, ddTheta1=None, ddTheta2=None, grid=True):
    """Plot the robot trajectory against time"""

    nrows = 2
    if dTheta1 is not None:
        assert dTheta2 is not None, "both dTheta1 and dTheta2 must be given, or neither"
        nrows += 2
    if ddTheta1 is not None:
        assert ddTheta2 is not None, "both ddTheta1 and ddTheta2 must be given, or neither"
        nrows += 2

    fig, ax = plt.subplots(nrows, 1, sharex=True, tight_layout=True)

    iplt = 0
    ax[iplt].plot(t, Theta1)
    ax[iplt].set_ylabel('Theta1')

    ax[iplt+1].plot(t, Theta2)
    ax[iplt+1].set_ylabel('Theta2')

    if dTheta1 is not None:
        iplt += 2
        ax[iplt].plot(t, dTheta1)
        ax[iplt].set_ylabel('dTheta1')

        ax[iplt+1].plot(t, dTheta2)
        ax[iplt+1].set_ylabel('dTheta2')

    if ddTheta1 is not None:
        iplt += 2
        ax[iplt].plot(t, ddTheta1)
        ax[iplt].set_ylabel('ddTheta1')

        ax[iplt+1].plot(t, ddTheta2)
        ax[iplt+1].set_ylabel('ddTheta2')

    ax[-1].set_xlabel('Time')

    if grid:
        for a in ax.flatten():
            a.grid()


def animate_robot(fig, ax, Theta1, Theta2, **kwargs):
    """Animate the robot"""

    interval = 50
    if 'interval' in kwargs:
        interval = kwargs.pop('interval')

    show_frame = True
    if 'show_frame' in kwargs:
        show_frame = kwargs.pop('show_frame')

    verbose_output = True
    if 'verbose_output' in kwargs:
        verbose_output = kwargs.pop('verbose_output')

    Theta1 = np.asarray(Theta1).flatten()
    Theta2 = np.asarray(Theta2).flatten()
    assert Theta1.shape[0] == Theta2.shape[0], "Theta1 and Theta2 should have the same length"
    num_frames = Theta1.shape[0]

    robot = plot_robot(ax, Theta1[0], Theta2[1], **kwargs)
    robot_plt = robot[0]
    circs = robot[1:4]

    txt = None
    if show_frame:
        factor = 0.9
        d = factor*(L1 + L2)
        txt = ax.text(-d, d, '', horizontalalignment='left')

    def init():
        out = [robot_plt] + circs
        if show_frame:
            out.append(txt)
        return out

    def update(frame):

        if verbose_output:
            print(f'frame {frame+1}/{num_frames}')
            print("  theta1 =", Theta1[frame])
            print("  theta2 =", Theta2[frame])

        if show_frame:
            txt.set_text(f'{frame+1}/{num_frames}')

        out = plot_robot(ax, Theta1[frame], Theta2[frame])
        if show_frame:
            out.append(txt)

        return out

    ani = FuncAnimation(
        fig, update,
        frames=range(num_frames),
        blit=True,
        init_func=init,
        interval=interval,
    )

    return ani


def main():

    # Plot robot
    fig, ax = plt.subplots(tight_layout=True)
    plot_robot(ax, np.deg2rad(30), np.deg2rad(40))

    # Animate robot
    Theta1 = np.deg2rad(np.linspace(10, 30))
    Theta2 = np.deg2rad(np.linspace(0, 90))
    fig, ax = plt.subplots(tight_layout=True)
    ani = animate_robot(fig, ax, Theta1, Theta2, verbose_output=True)

    plt.show()

if __name__ == '__main__':
    main()
