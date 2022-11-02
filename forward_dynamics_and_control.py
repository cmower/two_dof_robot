import numpy as np
from robot import FD, animate_robot, plt, gr

def main():

    n = 500
    dt = 0.01

    theta10, theta20 = np.deg2rad([0, 0])
    theta1g, theta2g = np.deg2rad([45, 90])

    def tau_0(theta1, theta2, dtheta1, dtheta2):
        """Apply zero torque"""
        return 0, 0

    def tau_gr(theta1, theta2, dtheta1, dtheta2):
        """Gravity compensation - i.e. hold robot at initial location"""
        return gr(theta10, theta20)

    def tau_ctrl(theta1, theta2, dtheta1, dtheta2):
        """Control - move robot to goal"""
        K = 100
        D = 20
        tau1 = K*(theta1g - theta1) - D*dtheta1
        tau2 = K*(theta2g - theta2) - D*dtheta2
        return tau1, tau2

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

    animate_robot(Theta1, Theta2, interval=dt*1000)

    plt.show()

if __name__ == '__main__':
    main()
