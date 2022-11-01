import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""

Model for a 2-dof plannar robot. This follows the notation and
equations set out in the online course "Modern Robotics" by Kevin
Lynch and Frank Park.

https://www.youtube.com/watch?v=jVu-Hijns70&list=PLggLP4f-rq02vX0OQQ5vrCxbJrzamYDfx

"""

def cos(theta):
    return np.cos(np.deg2rad(theta))

def sin(theta):
    return np.sin(np.deg2rad(theta))

g = -9.81
L1 = 1.
L2 = 1.
m1 = 1.5
m2 = 0.75
theta1_lim = [-135., 135.]
theta2_lim = [-135., 135.]

def x1(theta1, theta2):
    return L1*cos(theta1)

def y1(theta1, theta2):
    return L1*sin(theta1)

def dx1(theta1, theta2, dtheta1, dtheta2):
    return -L1*sin(theta1)*dtheta1

def dy1(theta1, theta2, dtheta1, dtheta2):
    return L1*cos(theta1)*dtheta1

def x2(theta1, theta2):
    return L1*cos(theta1) + L2*cos(theta1+theta2)

def y2(theta1, theta2):
    return L1*sin(theta1) + L2*sin(theta1+theta2)

def dx2(theta1, theta2, dtheta1, dtheta2):
    return (-L1*sin(theta1) - L2*sin(theta1+theta2))*dtheta1 + (-L2*sin(theta1+theta2))*dtheta2

def dy2(theta1, theta2, dtheta1, dtheta2):
    return (L1*cos(theta1) + L2*cos(theta1+theta2))*dtheta1 + (L2*cos(theta1+theta2))*dtheta2

def J(theta1, theta2):
    return np.array([
        [-L1*sin(theta1) - L2*sin(theta1+theta2), -L2*sin(theta1+theta2)],
        [L1*cos(theta1) + L2*cos(theta1+theta2), L2*cos(theta1+theta2)],
    ])

def K1(theta1, theta2, dtheta1, dtheta2):
    return 0.5*m1*(dx1(theta1, theta2, dtheta1, dtheta2)**2 + dy1(theta1, theta2, dtheta1, dtheta2)**2)

def K2(theta1, theta2, dtheta1, dtheta2):
    return 0.5*m2*(dx2(theta1, theta2, dtheta1, dtheta2)**2 + dy2(theta1, theta2, dtheta1, dtheta2)**2)

def P1(theta1, theta2):
    return m1*g*y1(theta1, theta2)

def P2(theta1, theta2):
    return m2*g*y2(theta1, theta2)

def Lag(theta1, theta2, dtheta1, dtheta2):
    return (K1(theta1, theta2, dtheta1, dtheta2) - P1(theta1, theta2, dtheta1, dtheta2)) + (K2(theta1, theta2, dtheta1, dtheta2) - P2(theta1, theta2, dtheta1, dtheta2))

def tau1(theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2):
    term1 = (m1*L1**2 + m2*(L1**2+2*L1*L2*cos(theta2)+L2**2))*ddtheta1 + m2*(L1*L2*cos(theta2)+L2**2)*ddtheta2
    term2 = -m2*L1*L2*sin(theta2)*(2*dtheta1*dtheta2+dtheta2**2)
    term3 = (m1+m2)*L1*g*cos(theta1) + m2*g*L2*cos(theta1 + theta2)
    return term1 + term2 + term3

def tau2(theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2):
    term1 = m2*(L1*L2*cos(theta2)+L2**2)*ddtheta1 + m2*L2**2*ddtheta2
    term2 = m2*L1*L2*dtheta1**2*sin(theta2)
    term3 = m2*g*L2*cos(theta1+theta2)
    return term1 + term2 + term3

def M(theta1, theta2):
    return np.array([
        [m1*L1**2 + m2*(L1**2+2*L1*L2*cos(theta2)+L2**2), m2*(L1*L2*cos(theta2)+L2**2)],
        [                   m2*(L1*L2*cos(theta2)+L2**2),                     m2*L2**2],
    ])

def C(theta1, theta2, dtheta1, dtheta2):
    return np.array([-m2*L1*L2*sin(theta2)*(2*dtheta1*dtheta2+dtheta2**2), m2*L1*L2*dtheta1**2*sin(theta2)])

def gr(theta1, theta2):
    return np.array([(m1+m2)*L1*g*cos(theta1) + m2*g*L2*cos(theta1 + theta2), m2*g*L2*cos(theta1+theta2)])

def plot_robot(ax, theta1, theta2, set_axis_lims=True, add_coord_axis=True, grid=True):

    robot_plt, = ax.plot(
        [0, x1(theta1, theta2), x2(theta1, theta2)],
        [0, y1(theta1, theta2), y2(theta1, theta2)],
        '-ok', markerfacecolor='blue', markeredgecolor='blue',
        zorder=3,
    )

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

    return robot_plt

def plot_trajectory(t, Theta1, Theta2, dTheta1=None, dTheta2=None, ddTheta1=None, ddTheta2=None, grid=True):
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

        
def animate_robot(Theta1, Theta2, **kwargs):

    interval = 50
    if 'interval' in kwargs:
        interval = kwargs.pop('interval')

    Theta1 = Theta1.flatten()
    Theta2 = Theta2.flatten()
    assert Theta1.shape[0] == Theta2.shape[0], "Theta1 and Theta2 should have the same length"

    fig, ax = plt.subplots(tight_layout=True)
    robot_plt = plot_robot(ax, Theta1[0], Theta2[1], **kwargs)

    def init():
        return (robot_plt,)

    def update(frame):
        return (plot_robot(ax, Theta1[frame], Theta2[frame]),)

    FuncAnimation(
        fig, update,
        frames=range(Theta1.shape[0]),
        blit=True,
        init_func=init,
        interval=interval,
    )


def main():

    # Plot robot
    fig, ax = plt.subplots(tight_layout=True)
    plot_robot(ax, 150, 100)

    # Animate robot
    Theta1 = np.linspace(10, 30)
    Theta2 = np.linspace(0, 90)
    animate_robot(Theta1, Theta2)

    plt.show()

if __name__ == '__main__':
    main()