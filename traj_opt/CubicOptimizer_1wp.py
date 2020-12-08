import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# plt.style.use('bmh')
# plt.rc('font', size=12)          # controls default text sizes
# plt.rc('axes', titlesize=20)     # fontsize of the axes title
# plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
# plt.rc('legend', fontsize=17)    # legend fontsize
# plt.rc('figure', titlesize=10)  # fontsize of the figure title


class CubicOptimizer_1wp:
    def __init__(self):
        self.q0 = []
        self.qw = []
        self.qf = []
        self.dqw = []
        self.max_vel = []
        self.max_acc = []
        self.t_step = []

    # calculate cubic coefficients of with minimim execution time
    def get_cubic_spline(self, q0, qf, v0, vf, tf_range=None):
        q0 = np.array(q0)
        qf = np.array(qf)
        v0 = np.array(v0)
        vf = np.array(vf)
        if tf_range is None:
            tf_range = [0.01, 5.0]
        tf = np.arange(start=tf_range[0], stop=tf_range[1], step=0.01).reshape(-1,1)
        a = (vf * tf + v0 * tf + 2 * q0 - 2 * qf) / tf ** 3
        b = (vf - v0) / (2 * tf) - 3 / 2 * a * tf
        c = v0
        d = q0

        # extreme values
        cond0 = (0 <= -b/(3*a)) & (-b/(3*a) <= tf)    # if extreme point is within the range (0~tf)
        cond1 = abs(-b**2/(3*a)+c) <= self.max_vel    # velocity constraint
        cond2 = abs(2*b) <= self.max_acc              # acceleration constraint at the left border
        cond3 = abs(6*a*tf + 2*b) <= self.max_acc     # acceleration constraint at the right border
        flag = ~(cond0 & ~cond1) & cond2 & cond3
        arg_min = min(np.argwhere(np.all(flag, axis=1)))[0]
        tf_min = tf[arg_min][0]
        coeff = np.concatenate((a[arg_min], b[arg_min], c, d)).reshape(-1,6)
        return coeff, tf_min

    def combine_trajectory(self, vw):
        v0 = np.zeros_like(vw)
        coeff1, tf1 = self.get_cubic_spline(q0=self.q0, qf=self.qw, v0=v0, vf=vw)  # 1st traj.
        vf = np.zeros_like(vw)
        coeff2, tf2 = self.get_cubic_spline(q0=self.qw, qf=self.qf, v0=vw, vf=vf)  # 2nd traj.

        t1 = np.arange(start=0.0, stop=tf1, step=self.t_step).reshape(-1,1)
        traj1 = coeff1[0]*t1**3 + coeff1[1]*t1**2 + coeff1[2]*t1 + coeff1[3]
        t2 = np.arange(start=0.0, stop=tf2, step=self.t_step).reshape(-1,1)
        traj2 = coeff2[0]*t2**3 + coeff2[1]*t2**2 + coeff2[2]*t2 + coeff2[3]

        traj = np.concatenate((traj1, traj2), axis=0)
        t = np.arange(start=0.0, stop=len(traj))*self.t_step
        return traj, t

    def get_cost(self, vw):
        v0 = np.zeros_like(vw)
        _, tf1 = self.get_cubic_spline(q0=self.q0, qf=self.qw, v0=v0, vf=vw)     # 1st traj.
        vf = np.zeros_like(vw)
        _, tf2 = self.get_cubic_spline(q0=self.qw, qf=self.qf, v0=vw, vf=vf)     # 2nd traj.
        return tf1+tf2

    def get_gradient(self, k):
        dk = 0.03   # perturbation
        cost_grad = (self.get_cost((k+dk)*self.dqw) - self.get_cost((k-dk)*self.dqw))/(2*dk)
        return cost_grad

    def optimize(self, q0, qw, qf, dqw, max_vel, max_acc, t_step=0.01, print_out=False, visualize=False):
        self.q0 = np.array(q0)
        self.qw = np.array(qw)
        self.qf = np.array(qf)
        self.dqw = np.array(dqw) / np.linalg.norm(dqw)    # normalized gradient (direction)
        self.max_vel = np.array(max_vel)
        self.max_acc = np.array(max_acc)
        self.t_step = np.array(t_step)

        # initial guess
        K = 0.0
        alpha = 0.02
        tf_mins = np.zeros(10)
        count = 0
        while True:
            f_grad = np.array(self.get_gradient(K))
            K = K - alpha*f_grad
            tf_min = self.get_cost(K*self.dqw)
            if print_out:
                print("K=", K)
                print("tf_min=", tf_min)
            tf_mins = np.insert(tf_mins, 0, tf_min)
            tf_mins = np.delete(tf_mins, -1)
            tf_mins_prev = np.insert(tf_mins, 0, tf_mins[0])
            tf_mins_prev = np.delete(tf_mins_prev, -1, axis=0)
            diff = abs(tf_mins - tf_mins_prev)
            if np.all(diff <= 0.02) and count > 10:
                traj, t = self.combine_trajectory(K*self.dqw)
                if visualize:
                    pos = traj
                    pos_prev = np.insert(pos, 0, pos[0], axis=0)
                    pos_prev = np.delete(pos_prev, -1, axis=0)
                    vel = (pos - pos_prev) / self.t_step
                    vel_prev = np.insert(vel, 0, vel[0], axis=0)
                    vel_prev = np.delete(vel_prev, -1, axis=0)
                    acc = (vel - vel_prev) / self.t_step
                    self.plot_joint(t, pos, vel, acc, t, pos, vel, acc)
                break
            count += 1
        return traj, t

    def plot_joint(self, t1, q1, q1_dot, q1_ddot, t2, q2, q2_dot, q2_ddot):
        # Create plot
        plt.figure(1)
        plt.title('Joint angle')
        ylabel = ['q1 (rad)', 'q2 (rad)', 'q3 (m)', 'q4 (rad)', 'q5 (rad)', 'q6 (rad)']
        for i in range(6):
            plt.subplot(610+i+1)
            plt.plot(t1, q1[:,i], 'b-', t2, q2[:,i], 'r-')
            plt.ylabel(ylabel[i])
        plt.xlabel('time (s)')

        plt.figure(2)
        plt.title('Joint velocity')
        ylabel = ['q1 (rad/s)', 'q2 (rad/s)', 'q3 (m/s)', 'q4 (rad/s)', 'q5 (rad/s)', 'q6 (rad/s)']
        for i in range(6):
            plt.subplot(610 + i + 1)
            plt.plot(t1, q1_dot[:, i], 'b-', t2, q2_dot[:, i], 'r-')
            plt.ylabel(ylabel[i])
        plt.xlabel('time (s)')

        plt.figure(3)
        plt.title('Joint acceleration')
        ylabel = ['q1 (rad/s^2)', 'q2 (rad/s^2)', 'q3 (m/s^2)', 'q4 (rad/s^2)', 'q5 (rad/s^2)', 'q6 (rad/s^2)']
        for i in range(6):
            plt.subplot(610 + i + 1)
            plt.plot(t1, q1_ddot[:, i], 'b-', t2, q2_ddot[:, i], 'r-')
            plt.ylabel(ylabel[i])
        plt.xlabel('time (s)')
        plt.show()

if __name__ == "__main__":
    opt = CubicOptimizer_1wp()
    # opt.search_min(q0, qw1, qw2, qf, qw1_grad, qw2_grad)