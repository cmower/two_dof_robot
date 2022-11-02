import numpy as np
from robot import FD, animate_robot, plt, gr, ID

"""

This script shows how the robot can be simulated using the forward
dynamics. Options are given to see how the robot response when (i)
zero torques are applied, (ii) the gravity term is used to compensate
for the effects of gravity, and (iii) a typical control approach is
used to move the robot to a goal configuration - comment the relevent
lines.

"""

def main():

    n = 500  # number of steps
    dt = 0.01  # time step

    theta10, theta20 = np.deg2rad([0, 0]) # initial robot configuration
    theta1g, theta2g = np.deg2rad([45, 90])  # goal robot configuration (when using tau_ctrl)

    def tau_0(theta1, theta2, dtheta1, dtheta2):
        """Apply zero torque"""
        return 0, 0

    def tau_gr(theta1, theta2, dtheta1, dtheta2):
        """Gravity compensation - i.e. hold robot at initial location"""
        return gr(theta10, theta20)

    def tau_ctrl(theta1, theta2, dtheta1, dtheta2):
        """Control (computed torque control) - move robot to goal"""
        K = 100  # stiffness gain
        D = 10  # damping gain
        return ID(theta1g, theta2g, 0, 0, K*(theta1g - theta1) - D*dtheta1, K*(theta2g - theta2) - D*dtheta2)

    Theta1 = np.zeros(n)
    Theta2 = np.zeros(n)
    Theta1[0] = theta10
    Theta2[0] = theta20

    dTheta1 = np.zeros(n)
    dTheta2 = np.zeros(n)

    ddTheta1 = np.zeros(n)
    ddTheta2 = np.zeros(n)

    for i in range(n-1):

        # tau1, tau2 = tau_0(Theta1[i], Theta2[i], dTheta1[i], dTheta2[i])
        # tau1, tau2 = tau_gr(Theta1[i], Theta2[i], dTheta1[i], dTheta2[i])
        tau1, tau2 = tau_ctrl(Theta1[i], Theta2[i], dTheta1[i], dTheta2[i])

        ddtheta1, ddtheta2 = FD(Theta1[i], Theta2[i], dTheta1[i], dTheta2[i], tau1, tau2)

        ddTheta1[i+1] = ddtheta1
        ddTheta1[i+1] = ddtheta2

        dTheta1[i+1] = dTheta1[i] + dt*ddtheta1
        dTheta2[i+1] = dTheta2[i] + dt*ddtheta2

        Theta1[i+1] = Theta1[i] + dt*dTheta1[i]
        Theta2[i+1] = Theta2[i] + dt*dTheta2[i]

    fig, ax = plt.subplots(tight_layout=True)
    animate_robot(fig, ax, Theta1, Theta2, interval=dt*1000)

    plt.show()

if __name__ == '__main__':
    main()
