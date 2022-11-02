import time
import numpy as np
from matplotlib.patches import Circle
from scipy.optimize import minimize, NonlinearConstraint

try:
    # https://github.com/mechmotum/cyipopt
    from cyipopt import minimize_ipopt
    ipopt_available = True
except ImportError:
    ipopt_available = False

from robot import x2, y2, theta1_lim, theta2_lim, animate_robot, plt, plot_trajectory, robot_joint_radius, x1, y1

"""

This script shows how to formulate a trajectory optimization problem
for planning. The goal is to find a plan from an initial configuration
to a goal state in a given duration whilst minimizing joint velocity
and acceleration. The goal state is given by a goal end-effector
position. Optionally, an obstacle can be included.

"""

def plan_joint_trajectory_to_goal(theta10, theta20, x2g, y2g, duration, n, method, wx=1e3, wy=1e3, wd1=0.01, wd2=0.01, wdd1=1, wdd2=1, disp=False, animate=True, plot_traj=True, add_obs=True, obs_pos=[0.5, 1.5], obs_rad=0.1):

    t = np.linspace(0, duration, n)  # time evolution
    dt = t[1]  # time step

    def cost(X):
        """Cost function: (i) reach end-effector goal, (ii) minimize velocty, (iii) minimize acceleration"""
        traj = X.reshape(4, n)
        Theta1 = traj[0,:]
        Theta2 = traj[1,:]
        dTheta1 = traj[2,:]
        dTheta2 = traj[3,:]
        ddTheta1 = (dTheta1[1:] - dTheta1[:-1])/dt
        ddTheta2 = (dTheta2[1:] - dTheta2[:-1])/dt
        dx = x2(Theta1[-1], Theta2[-1]) - x2g
        dy = y2(Theta1[-1], Theta2[-1]) - y2g
        return wx*dx**2 + wx*dy**2 + wd1*np.sum(dTheta1**2) + wd2*np.sum(dTheta2**2) + wdd1*np.sum(ddTheta1**2) + wdd2*np.sum(ddTheta2**2)

    def dynamics(X):
        """Model system dynamics as equality constraints"""
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
        """Model initial configuration as equality constraints"""
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
        """Model final configuration as equality constraints"""
        traj = X.reshape(4, n)
        Theta1 = traj[0,:]
        Theta2 = traj[1,:]
        dTheta1 = traj[2,:]
        dTheta2 = traj[3,:]
        ddTheta1 = (dTheta1[1:] - dTheta1[:-1])/dt
        ddTheta2 = (dTheta2[1:] - dTheta2[:-1])/dt
        return np.array([dTheta1[-1], dTheta2[-1], ddTheta1[-1], ddTheta2[-1]])

    def joint_limits(X):
        """Model joint limits as inequality constraints"""
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

    def obstacle_avoidance(X):
        """Model obstacle avoidance as inequality constraint"""
        traj = X.reshape(4, n)
        Theta1 = traj[0,:]
        Theta2 = traj[1,:]
        diff = np.zeros((2*5, n))

        p1 = np.stack((x1(Theta1, Theta2), y1(Theta1, Theta2)))
        p2 = np.stack((x2(Theta1, Theta2), y2(Theta1, Theta2)))

        dr2 = p2 - p1

        p11 = p1 + 0.25*dr2
        p12 = p1 + 0.5*dr2
        p13 = p1 + 0.75*dr2

        diff[0,:] = (p1[0,:] - obs_pos[0])**2 - (obs_rad + robot_joint_radius)**2
        diff[1,:] = (p1[1,:] - obs_pos[1])**2 - (obs_rad + robot_joint_radius)**2
        diff[2,:] = (p2[0,:] - obs_pos[0])**2 - (obs_rad + robot_joint_radius)**2
        diff[3,:] = (p2[1,:] - obs_pos[1])**2 - (obs_rad + robot_joint_radius)**2
        diff[4,:] = (p11[0,:] - obs_pos[0])**2 - (obs_rad + robot_joint_radius)**2
        diff[5,:] = (p11[1,:] - obs_pos[1])**2 - (obs_rad + robot_joint_radius)**2
        diff[6,:] = (p12[0,:] - obs_pos[0])**2 - (obs_rad + robot_joint_radius)**2
        diff[7,:] = (p12[1,:] - obs_pos[1])**2 - (obs_rad + robot_joint_radius)**2
        diff[8,:] = (p13[0,:] - obs_pos[0])**2 - (obs_rad + robot_joint_radius)**2
        diff[9,:] = (p13[1,:] - obs_pos[1])**2 - (obs_rad + robot_joint_radius)**2

        diff[4:8,:]*=0

        return diff.flatten()

    traj0 = np.zeros((4, n))
    if not add_obs:
        traj0[0,:] = theta10*np.ones(n)
        traj0[1,:] = theta20*np.ones(n)
    else:
        # When using obstacle, above initial guess gets stuck in a
        # local minima, solve the problem without the obstacle and use
        # the solution as a warm start
        Theta10, Theta20, dTheta10, dTheta20 = \
            plan_joint_trajectory_to_goal(theta10, theta20, x2g, y2g, duration, n, 'SLSQP', wx=100, wy=100, wd1=1, wd2=1, wdd1=0.1, wdd2=0.1, animate=False, disp=False, add_obs=False, plot_traj=False)
        traj0[0,:] = Theta10
        traj0[1,:] = Theta20
        traj0[2,:] = dTheta10
        traj0[3,:] = dTheta20
    X0 = traj0.flatten()

    minimize_ = minimize
    
    constraints = []
    if method == 'trust-constr':
        constraints.append(NonlinearConstraint(dynamics, lb=np.zeros(2*(n-1)), ub=np.zeros(2*(n-1))))
        constraints.append(NonlinearConstraint(initial_config, lb=np.zeros(6), ub=np.zeros(6)))
        constraints.append(NonlinearConstraint(final_config, lb=np.zeros(4), ub=np.zeros(4)))
        constraints.append(NonlinearConstraint(joint_limits, lb=np.zeros(4*n), ub=np.inf))
        if add_obs:
            constraints.append(NonlinearConstraint(obstacle_avoidance, lb=np.zeros(2*5*n), ub=np.inf))

    elif method == 'SLSQP':
        constraints.append({'type': 'eq', 'fun': dynamics})
        constraints.append({'type': 'eq', 'fun': initial_config})
        constraints.append({'type': 'eq', 'fun': final_config})
        constraints.append({'type': 'ineq', 'fun': joint_limits})
        if add_obs:
            constraints.append({'type': 'ineq', 'fun': obstacle_avoidance})

    elif method == 'COBYLA':
        constraints.append({'type': 'ineq', 'fun': dynamics})
        constraints.append({'type': 'ineq', 'fun': lambda X: -dynamics(X)})
        constraints.append({'type': 'ineq', 'fun': initial_config})
        constraints.append({'type': 'ineq', 'fun': lambda X: -initial_config(X)})
        constraints.append({'type': 'ineq', 'fun': final_config})
        constraints.append({'type': 'ineq', 'fun': lambda X: -final_config(X)})
        constraints.append({'type': 'ineq', 'fun': joint_limits})
        if add_obs:
            constraints.append({'type': 'ineq', 'fun': obstacle_avoidance})

    elif method == 'ipopt' and ipopt_available:
        minimize_ = minimize_ipopt
        if disp:
            disp = 5
        constraints.append({'type': 'eq', 'fun': dynamics})
        constraints.append({'type': 'eq', 'fun': initial_config})
        constraints.append({'type': 'eq', 'fun': final_config})
        constraints.append({'type': 'ineq', 'fun': joint_limits})
        if add_obs:
            constraints.append({'type': 'ineq', 'fun': obstacle_avoidance})        

    else:
        raise ValueError(f"method '{method}' is not supported")


    res = minimize_(cost, X0, method=method, constraints=constraints, options={'disp': disp})
    success = res.success

    print(f"Did {method} solve the problem?", 'yes' if success else 'no')

    traj = res.x.reshape(4, n)
    Theta1 = traj[0,:]
    Theta2 = traj[1,:]
    dTheta1 = traj[2,:]
    dTheta2 = traj[3,:]

    if animate:
        fig, ax = plt.subplots(tight_layout=True)
        if add_obs:
            ax.add_patch(Circle(obs_pos, radius=obs_rad, color='k'))
        animate_robot(fig, ax, Theta1, Theta2, interval=dt*1000)

    if plot_traj:
        ddTheta1 = np.gradient(dTheta1, t)
        ddTheta2 = np.gradient(dTheta2, t)
        plot_trajectory(t, Theta1, Theta2, dTheta1=dTheta1, dTheta2=dTheta2, ddTheta1=ddTheta1, ddTheta2=ddTheta2)

    plt.show()

    return Theta1, Theta2, dTheta1, dTheta2

def main():

    # Plan joint trajectory
    theta10 = np.deg2rad(30)  # initial position for joint 1
    theta20 = np.deg2rad(40)  # initial position for joint 2
    x2g = -1  # goal end-effector position in x-axis
    y2g = 0.5  # goal end-effector position in y-axis
    duration = 5  # duration of trajectory
    n = 30 # number of steps
    method = 'SLSQP' # solver used to solve trajectory optimization problem

    Theta1, Theta2, dTheta1, dTheta2 = \
        plan_joint_trajectory_to_goal(theta10, theta20, x2g, y2g, duration, n, method, wx=100, wy=100, wd1=0.1, wd2=0.1, wdd1=0.01, wdd2=0.01, animate=True, disp=True)

if __name__ == '__main__':
    main()
