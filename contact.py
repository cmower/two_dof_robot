import numpy as np
from robot import FD, animate_robot, plt, gr, J

def main():

    n = 1000
    dt = 0.01

    theta10, theta20 = np.deg2rad([45, 90])

    Theta1 = np.zeros(n)
    Theta2 = np.zeros(n)
    Theta1[0] = theta10
    Theta2[0] = theta20

    dTheta1 = np.zeros(n)
    dTheta2 = np.zeros(n)

    ddTheta1 = np.zeros(n)
    ddTheta2 = np.zeros(n)

    Fext = np.array([30, 0])

    def tau_ctrl(theta1, theta2, dtheta1, dtheta2):
        K = 100
        D = 20
        tau1 = K*(theta10 - theta1) - D*dtheta1
        tau2 = K*(theta20 - theta2) - D*dtheta2
        return tau1, tau2

    for i in range(n-1):

        tau1, tau2 = tau_ctrl(Theta1[i], Theta2[i], dTheta1[i], dTheta2[i])

        if 200 < i < 250:
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
