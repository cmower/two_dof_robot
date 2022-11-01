import time
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from robot import x2, y2, theta1_lim, theta2_lim, animate_robot, plt, plot_trajectory

def plan_joint_trajectory_to_goal(theta10, theta20, x2g, y2g, duration, n, method, wx=1e6, wy=1e6, wdx=0.01, wdy=0.01, wddx=1, wddy=1, disp=False, animate=True, plot_traj=True):

    t = np.linspace(0, duration, n)
    dt = t[1]

    def cost(X):
        traj = X.reshape(4, n)
        Theta1 = traj[0,:]
        Theta2 = traj[1,:]
        dTheta1 = traj[2,:]
        dTheta2 = traj[3,:]
        ddTheta1 = (dTheta1[1:] - dTheta1[:-1])/dt
        ddTheta2 = (dTheta2[1:] - dTheta2[:-1])/dt
        dx = x2(Theta1[-1], Theta2[-1]) - x2g
        dy = y2(Theta1[-1], Theta2[-1]) - y2g
        return wx*dx**2 + wx*dy**2 + wdx*np.sum(dTheta1**2) + wdy*np.sum(dTheta2**2) + wddx*np.sum(ddTheta1**2) + wddy*np.sum(ddTheta2**2)

    def dynamics(X):
        traj = X.reshape(4, n)
        Theta1 = traj[0,:]
        Theta2 = traj[1,:]
        dTheta1 = traj[2,:]
        dTheta2 = traj[3,:]

        diff = np.zeros((2, n-1))
        diff[0,:] = Theta1[:-1] + dt*dTheta1[:-1] - Theta1[1:]
        diff[1,:] = Theta2[:-1] + dt*dTheta2[:-1] - Theta2[1:]

        return diff.flatten()

    def initial_config(X):
        traj = X.reshape(4, n)
        Theta1 = traj[0,:]
        Theta2 = traj[1,:]
        dTheta1 = traj[2,:]
        dTheta2 = traj[3,:]
        ddTheta1 = (dTheta1[1:] - dTheta1[:-1])/dt
        ddTheta2 = (dTheta2[1:] - dTheta2[:-1])/dt
        return np.array([
            Theta1[0] - theta10,
            Theta2[0] - theta20,
            dTheta1[0],
            dTheta2[0],
            ddTheta1[0],
            ddTheta2[0],
        ])

    def final_config(X):
        traj = X.reshape(4, n)
        Theta1 = traj[0,:]
        Theta2 = traj[1,:]
        dTheta1 = traj[2,:]
        dTheta2 = traj[3,:]
        ddTheta1 = (dTheta1[1:] - dTheta1[:-1])/dt
        ddTheta2 = (dTheta2[1:] - dTheta2[:-1])/dt
        return np.array([dTheta1[-1], dTheta2[-1], ddTheta1[-1], ddTheta2[-1]])

    def joint_limits(X):
        traj = X.reshape(4, n)
        Theta1 = traj[0,:]
        Theta2 = traj[1,:]
        dTheta1 = traj[2,:]
        dTheta2 = traj[3,:]

        diff = np.zeros((4, n))
        diff[0,:] = Theta1 - theta1_lim[0]
        diff[1,:] = theta1_lim[1] - Theta1
        diff[2,:] = Theta2 - theta2_lim[0]
        diff[3,:] = theta2_lim[1] - Theta2

        return diff.flatten()

    traj0 = np.zeros((4, n))
    traj0[0,:] = theta10*np.ones(n)
    traj0[1,:] = theta20*np.ones(n)
    X0 = traj0.flatten()

    constraints = []
    if method == 'trust-constr':
        constraints.append(NonlinearConstraint(dynamics, lb=np.zeros(2*(n-1)), ub=np.zeros(2*(n-1))))
        constraints.append(NonlinearConstraint(initial_config, lb=np.zeros(6), ub=np.zeros(6)))
        constraints.append(NonlinearConstraint(final_config, lb=np.zeros(4), ub=np.zeros(4)))
        constraints.append(NonlinearConstraint(joint_limits, lb=np.zeros(4*n), ub=np.inf))

    elif method == 'SLSQP':
        constraints.append({'type': 'eq', 'fun': dynamics})
        constraints.append({'type': 'eq', 'fun': initial_config})
        constraints.append({'type': 'eq', 'fun': final_config})
        constraints.append({'type': 'ineq', 'fun': joint_limits})

    elif method == 'COBYLA':
        constraints.append({'type': 'ineq', 'fun': dynamics})
        constraints.append({'type': 'ineq', 'fun': lambda X: -dynamics(X)})
        constraints.append({'type': 'ineq', 'fun': initial_config})
        constraints.append({'type': 'ineq', 'fun': lambda X: -initial_config(X)})
        constraints.append({'type': 'ineq', 'fun': final_config})
        constraints.append({'type': 'ineq', 'fun': lambda X: -final_config(X)})
        constraints.append({'type': 'ineq', 'fun': joint_limits})

    else:
        raise ValueError(f"method '{method}' is not supported")

    res = minimize(cost, X0, method=method, constraints=constraints, options={'disp': disp})

    traj = res.x.reshape(4, n)
    Theta1 = traj[0,:]
    Theta2 = traj[1,:]
    dTheta1 = traj[2,:]
    dTheta2 = traj[3,:]

    if animate:
        animate_robot(Theta1, Theta2, interval=dt*1000)

    if plot_traj:
        ddTheta1 = np.gradient(dTheta1, t)
        ddTheta2 = np.gradient(dTheta2, t)
        plot_trajectory(t, Theta1, Theta2, dTheta1=dTheta1, dTheta2=dTheta2, ddTheta1=ddTheta1, ddTheta2=ddTheta2)

    plt.show()

    return Theta1, Theta2, dTheta1, dTheta2

def main():

    # Plan joint trajectory
    theta10 = 30
    theta20 = 40
    x2g = -1
    y2g = 0.5
    duration = 4
    n = 50
    method = 'SLSQP'

    Theta1, Theta2, dTheta1, dTheta2 = \
        plan_joint_trajectory_to_goal(theta10, theta20, x2g, y2g, duration, n, method, animate=True, disp=True)

if __name__ == '__main__':
    main()
