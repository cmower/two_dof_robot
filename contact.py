import numpy as np
from robot import FD, animate_robot, plt, gr, J

"""

This script demonstrates how a robot controller responds when an
external force is applied at the end-effector. The controller attempts
to maintain the initial robot position, and at the 200th iteration, an
external force is temporarily applied for a short amount of time.

"""

def main():

    n = 1000  # number of steps
    dt = 0.01  # time step

    theta10, theta20 = np.deg2rad([45, 90])  # initial robot configuration

    Theta1 = np.zeros(n)
    Theta2 = np.zeros(n)
    Theta1[0] = theta10
    Theta2[0] = theta20

    dTheta1 = np.zeros(n)
    dTheta2 = np.zeros(n)

    ddTheta1 = np.zeros(n)
    ddTheta2 = np.zeros(n)

    Fext = np.array([30, 0])  # external contact force at end-effector
    i_start = 200  # iteration that the external contact force is applied from
    i_end = 250  # iteration that the external contact force is applied till

    def tau_ctrl(theta1, theta2, dtheta1, dtheta2):
        K = 100  # stiffness gain
        D = 20  # damping gain
        tau1 = K*(theta10 - theta1) - D*dtheta1
        tau2 = K*(theta20 - theta2) - D*dtheta2
        return tau1, tau2

    for i in range(n-1):

        tau1, tau2 = tau_ctrl(Theta1[i], Theta2[i], dTheta1[i], dTheta2[i])

        if i_start < i < i_end:
            # apply external contact force at end-effector
            tau_ext = J(Theta1[i], Theta2[i]).T@Fext
            tau1 += tau_ext[0]
            tau2 += tau_ext[1]

        ddtheta1, ddtheta2 = FD(Theta1[i], Theta2[i], dTheta1[i], dTheta2[i], tau1, tau2)

        ddTheta1[i+1] = ddtheta1
        ddTheta1[i+1] = ddtheta2

        dTheta1[i+1] = dTheta1[i] + dt*ddtheta1
        dTheta2[i+1] = dTheta2[i] + dt*ddtheta2

        Theta1[i+1] = Theta1[i] + dt*dTheta1[i]
        Theta2[i+1] = Theta2[i] + dt*dTheta2[i]

    animate_robot(Theta1, Theta2, interval=dt*1000)

    plt.show()

if __name__ == '__main__':
    main()
