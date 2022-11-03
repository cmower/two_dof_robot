import os
import time
import numpy as np
import pandas as pd
from robot import FD, animate_robot, randtheta, gr, plt, ID

def main(animate=True, Ntrials=100, trial_duration=10):

    theta10, theta20 = np.deg2rad([0, 0])
    dt = 0.01

    theta1, theta2 = theta10, theta20
    dtheta1, dtheta2 = 0, 0
    ddtheta1, ddtheta2 = 0, 0
    tau1, tau2 = gr(theta1, theta2)

    def tau(theta1, theta2, dtheta1, dtheta2, theta1t, theta2t):
        """Control (computed torque control) - move robot to goal"""
        K = 200  # stiffness gain
        D = 20  # damping gain
        return ID(theta1t, theta2t, 0, 0, K*(theta1t - theta1) - D*dtheta1, K*(theta2t - theta2) - D*dtheta2)

    Theta1, Theta2 = [theta1], [theta2]
    dTheta1, dTheta2 = [dtheta1], [dtheta2]
    ddTheta1, ddTheta2 = [ddtheta1], [ddtheta2]
    Tau1, Tau2 = [tau1], [tau2]

    for i in range(Ntrials):

        theta1g, theta2g = randtheta()
        t = 0.

        while t < trial_duration:

            alpha = t/trial_duration

            theta1t = (1-alpha)*theta10 + alpha*theta1g
            theta2t = (1-alpha)*theta20 + alpha*theta2g

            tau1, tau2 = tau(theta1, theta2, dtheta1, dtheta2, theta2t, theta2t)

            ddtheta1, ddtheta2 = FD(theta1, theta2, dtheta2, dtheta2, tau1, tau2, apply_friction=False)

            dtheta1 += dt*ddtheta1
            dtheta2 += dt*ddtheta2

            theta1 += dt*dtheta1
            theta2 += dt*dtheta2

            Theta1.append(theta1)
            Theta2.append(theta2)
            dTheta1.append(dtheta1)
            dTheta2.append(dtheta2)
            ddTheta1.append(ddtheta1)
            ddTheta2.append(ddtheta2)
            Tau1.append(tau1)
            Tau2.append(tau2)

            t += dt

        theta10, theta20 = theta1, theta2

    datadir = os.path.join('.', 'data')
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    stamp = time.time_ns()
    filename = os.path.join(datadir, f'learn_id_data_{stamp}.csv')
    df = pd.DataFrame({'theta1': Theta1, 'theta2': Theta2, 'dtheta1': dTheta1, 'dtheta2': dTheta2, 'ddtheta1': ddTheta1, 'ddtheta2': ddTheta2, 'tau1': Tau1, 'tau2': Tau2})
    df.to_csv(filename)
    print("Saved", filename)

    if animate:
        fig, ax = plt.subplots(tight_layout=True)
        animate_robot(fig, ax, Theta1, Theta2, interval=dt*1000)

    plt.show()

    return filename

if __name__ == '__main__':
    main()
