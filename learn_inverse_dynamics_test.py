import sys
import pickle
import numpy as np
from robot import FD, animate_robot, plt, gr, J, ID, Fr

def main(filename, verbose_output=True, animate=True, plot_cmp=True):

    with open(filename, 'rb') as f:
        regr = pickle.load(f)

    def predict_tau_ext(theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2, tau1, tau2):
        x = np.array([theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2])
        return regr.predict(x.reshape(1, -1)).flatten() - np.array([tau1, tau2]) #+ Fr(dtheta1, dtheta2)

    n = 500  # number of steps
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

    Tau1 = np.zeros(n)
    Tau2 = np.zeros(n)
    tau1, tau2 = gr(theta10, theta20)
    Tau1[0] = tau1
    Tau2[0] = tau2

    TauExt1 = np.zeros(n)
    TauExt2 = np.zeros(n)

    PTauExt1 = np.zeros(n)
    PTauExt2 = np.zeros(n)

    Fext = np.array([30, 0])  # external contact force at end-effector
    i_start = 200  # iteration that the external contact force is applied from
    i_end = 250  # iteration that the external contact force is applied till

    t = 0.
    T = [t]

    def tau_ctrl(theta1, theta2, dtheta1, dtheta2):
        """Control (computed torque control) - move robot to goal"""
        K = 200  # stiffness gain
        D = 20  # damping gain
        return ID(theta10, theta20, 0, 0, K*(theta10 - theta1) - D*dtheta1, K*(theta20 - theta2) - D*dtheta2)

    for i in range(n-1):

        tau1, tau2 = tau_ctrl(Theta1[i], Theta2[i], dTheta1[i], dTheta2[i])
        Tau1[i+1] = tau1
        Tau2[i+1] = tau2

        # apply external contact force at end-effector
        tau_ext = np.zeros(2)
        if i_start < i < i_end:
            tau_ext = J(Theta1[i], Theta2[i]).T@Fext

        TauExt1[i+1] = tau_ext[0]
        TauExt2[i+1] = tau_ext[1]

        tau1 += tau_ext[0]
        tau2 += tau_ext[1]

        ddtheta1, ddtheta2 = FD(Theta1[i], Theta2[i], dTheta1[i], dTheta2[i], tau1, tau2, apply_noise=False, eps=0.5, apply_friction=False)

        ddTheta1[i+1] = ddtheta1
        ddTheta1[i+1] = ddtheta2

        dTheta1[i+1] = dTheta1[i] + dt*ddtheta1
        dTheta2[i+1] = dTheta2[i] + dt*ddtheta2

        Theta1[i+1] = Theta1[i] + dt*dTheta1[i+1]
        Theta2[i+1] = Theta2[i] + dt*dTheta2[i+1]

        ptau_ext1, ptau_ext2 = predict_tau_ext(Theta1[i], Theta2[i], dTheta1[i], dTheta2[i], ddtheta1, ddtheta2, Tau1[i], Tau2[i])
        PTauExt1[i+1] = ptau_ext1
        PTauExt2[i+1] = ptau_ext2

        t += dt
        T.append(t)

    if plot_cmp:
        fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
        ax[0].plot(T, TauExt1, label='tau_ext1')
        ax[0].plot(T, PTauExt1, label='ptau_ext1')

        ax[1].plot(T, TauExt2, label='tau_ext2')
        ax[1].plot(T, PTauExt2, label='ptau_ext2')

        ax[-1].set_xlabel('Time')
        ax[0].set_ylabel('Joint 1')
        ax[1].set_ylabel('Joint 2')

        for a in ax.flatten():
            a.grid()
            a.legend()

        if animate:
            fig, ax = plt.subplots(tight_layout=True)
            animate_robot(fig, ax, Theta1, Theta2, interval=dt*1000, verbose_output=verbose_output)    

    plt.show()

    return {
        't': T,        
        'theta1': Theta1,
        'theta2': Theta2,
        'dtheta1': dTheta1,
        'dtheta2': dTheta2,
        'ddtheta1': ddTheta1,
        'ddtheta2': ddTheta2,
        'tau1': Tau1,
        'tau2': Tau2,
        'tauext1': TauExt1,
        'tauext2': TauExt2,
        'ptauext1': PTauExt1,
        'ptauext2': PTauExt2,
    }

if __name__ == '__main__':
    try:
        filename = sys.argv[1]
    except IndexError:
        print("Filename must be given.")
        sys.exit(0)
    main(filename)
