import numpy as np
import time
import matplotlib.pyplot as plt
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics

plt.style.use('seaborn-whitegrid')
# plt.style.use('bmh')
# plt.rc('font', size=12)          # controls default text sizes
# plt.rc('axes', titlesize=20)     # fontsize of the axes title
# plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
# plt.rc('legend', fontsize=17)    # legend fontsize
# plt.rc('figure', titlesize=10)  # fontsize of the figure title


class CubicOptimizer_2wp:
    def __init__(self, max_vel, max_acc, t_step=0.01, print_out=False, visualize=False):
        self.q0 = []
        self.qw1 = []
        self.qw2 = []
        self.qf = []
        self.dqw1 = []
        self.dqw2 = []
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.t_step = t_step
        self.print_out = print_out
        self.visualize = visualize

    # calculate cubic coefficients of with minimim execution time
    def get_cubic_spline(self, q0, qf, v0, vf, tf_range=None):
        assert(np.all(np.abs(v0) <= self.max_vel))
        assert(np.all(np.abs(vf) <= self.max_vel))
        q0 = np.array(q0)
        qf = np.array(qf)
        v0 = np.array(v0)
        vf = np.array(vf)
        if tf_range is None:
            tf_range = np.array([0.01, 4.0, 0.01])
        tf = np.arange(start=tf_range[0], stop=tf_range[1], step=tf_range[2]).reshape(-1, 1)
        a = (vf * tf + v0 * tf + 2 * q0 - 2 * qf) / tf ** 3
        b = (vf - v0) / (2 * tf) - 3 / 2 * a * tf
        c = v0
        d = q0

        # extreme values
        cond0 = (0 <= -b/(3*a)) & (-b/(3*a) <= tf)  # if extreme point is within the range (0~tf)
        cond1 = abs(-b**2/(3*a)+c) <= self.max_vel       # velocity constraint
        cond2 = abs(2*b) <= self.max_acc                 # acceleration constraint at the left border
        cond3 = abs(6*a*tf + 2*b) <= self.max_acc        # acceleration constraint at the right border
        flag = ~(cond0 & ~cond1) & cond2 & cond3
        arg_min = min(np.argwhere(np.all(flag, axis=1)))[0]
        tf_min = tf[arg_min][0]
        coeff = np.concatenate((a[arg_min], b[arg_min], c, d)).reshape(-1,6)
        return coeff, tf_min

    def combine_trajectory(self, vw1, vw2):
        v0 = np.zeros_like(vw1)
        coeff1, tf1 = self.get_cubic_spline(q0=self.q0, qf=self.qw1, v0=v0, vf=vw1, tf_range=[0.01, 2.0, self.t_step])  # 1st traj.
        coeff2, tf2 = self.get_cubic_spline(q0=self.qw1, qf=self.qw2, v0=vw1, vf=vw2, tf_range=[0.01, 2.0, self.t_step])  # 2nd traj.
        vf = np.zeros_like(vw1)
        coeff3, tf3 = self.get_cubic_spline(q0=self.qw2, qf=self.qf, v0=vw2, vf=vf, tf_range=[0.01, 2.0, self.t_step])  # 3rd traj.

        t1 = np.arange(start=0.0, stop=tf1, step=self.t_step).reshape(-1,1)
        traj1 = coeff1[0]*t1**3 + coeff1[1]*t1**2 + coeff1[2]*t1 + coeff1[3]
        t2 = np.arange(start=0.0, stop=tf2, step=self.t_step).reshape(-1,1)
        traj2 = coeff2[0]*t2**3 + coeff2[1]*t2**2 + coeff2[2]*t2 + coeff2[3]
        t3 = np.arange(start=0.0, stop=tf3, step=self.t_step).reshape(-1,1)
        traj3 = coeff3[0]*t3**3 + coeff3[1]*t3**2 + coeff3[2]*t3 + coeff3[3]

        traj = np.concatenate((traj1, traj2, traj3), axis=0)
        t = np.arange(start=0.0, stop=len(traj))*self.t_step
        return traj, t

    def get_cost(self, vw1, vw2):
        v0 = np.zeros_like(vw1)
        _, tf1 = self.get_cubic_spline(q0=self.q0, qf=self.qw1, v0=v0, vf=vw1, tf_range=[0.3, 2.0, 0.02])  # 1st traj.
        _, tf2 = self.get_cubic_spline(q0=self.qw1, qf=self.qw2, v0=vw1, vf=vw2, tf_range=[0.3, 2.0, 0.02])  # 2nd traj.
        vf = np.zeros_like(vw1)
        _, tf3 = self.get_cubic_spline(q0=self.qw2, qf=self.qf, v0=vw2, vf=vf, tf_range=[0.3, 2.0, 0.02])  # 3rd traj.
        return tf1+tf2+tf3

    def get_gradient(self, k1, k2):
        dk1 = 0.15   # perturbation
        dk2 = 0.15
        cost_grad_k1 = (self.get_cost((k1+dk1)*self.dqw1, k2*self.dqw2) - self.get_cost((k1-dk1)*self.dqw1, k2*self.dqw2))/(2*dk1)
        cost_grad_k2 = (self.get_cost(k1*self.dqw1, (k2+dk2)*self.dqw2) - self.get_cost(k1*self.dqw1, (k2-dk2)*self.dqw2))/(2*dk2)
        return [cost_grad_k1, cost_grad_k2]

    def optimize(self, q0, qw1, qw2, qf):
        self.q0 = np.array(q0).squeeze()
        self.qw1 = np.array(qw1).squeeze()
        self.qw2 = np.array(qw2).squeeze()
        self.qf = np.array(qf).squeeze()
        self.max_vel = np.array(self.max_vel).squeeze()
        self.max_acc = np.array(self.max_acc).squeeze()
        self.t_step = np.array(self.t_step).squeeze()

        # Calculate dqw1, dqw2
        J1 = dvrkKinematics.jacobian(qw1)
        J2 = dvrkKinematics.jacobian(qw2)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw1 = np.linalg.inv(J1).dot(dvw)
        dqw2 = np.linalg.inv(J2).dot(dvw)
        self.dqw1 = np.array(dqw1).squeeze() / np.linalg.norm(dqw1)  # normalized gradient (direction)
        self.dqw2 = np.array(dqw2).squeeze() / np.linalg.norm(dqw2)

        # initial guess
        K = np.array([0.0, 0.0])
        alpha = 0.06
        tf_mins = []
        Ks = []
        count = 0
        while True:
            f_grad = np.array(self.get_gradient(K[0], K[1]))
            K = K - alpha*f_grad
            tf_min = self.get_cost(K[0]*self.dqw1, K[1]*self.dqw2)
            tf_mins.append(tf_min)
            Ks.append(K)
            if self.print_out:
                print (tf_min, K, count)
            if count > 50:
                Ks = np.array(Ks)
                tf_mins = np.array(tf_mins)
                arg = np.argmin(tf_mins)
                if self.print_out:
                    print("selected: ", tf_mins[arg], Ks[arg])
                traj, t = self.combine_trajectory(Ks[arg][0]*self.dqw1, Ks[arg][1]*self.dqw2)
                if self.visualize:
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
    opt = CubicOptimizer_2wp()
    # opt.search_min(q0, qw1, qw2, qf, qw1_grad, qw2_grad)
