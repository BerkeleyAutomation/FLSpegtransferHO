import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import time


class PegMotionOptimizer:
    def __init__(self):
        pass

    def define_QP_matrix1(self, p0, pw, pf, max_vel, max_acc, horizon, nb_waypoint):
        # define Q, c, A_ub, b_ub, A_eq, b_eq
        # such that x = [x0, ..., x_(H-1) | y0, ..., y_(H-1) | z0, ..., z_(H-1) | r0, ..., r_(H-1) | dt1**2, dt2**2] (4*H + 2)
        #          Q = [[ 0,  0,  0,  0, ...,  0,  0,  0,  0], (4*H+4) x (4*H+4)
        #               [ 0,  0,  0,  0, ...,  0,  0,  0,  0],
        #               [                ...,               ],
        #               [ 0,  0,  0,  0, ...,  n,  0,  0,  0],
        #               [ 0,  0,  0,  0, ...,  0,(H-n),0,  0],
        #               [ 0,  0,  0,  0, ...,  0,  0,  0,  0],
        #               [ 0,  0,  0,  0, ...,  0,  0,  0,  0]])
        #           c = [0, ..., 0, n, H-n] (4*H + 2)
        #        A_ub = [1, -2,  1,  0, ...,  0,  0,  0, 0 | 0, 0, -a, 0]  (4*2*(H-2)+4*2*(H-1)) x (4*H+4)
        #               [0,  1, -2,  1, ...,  0,  0,  0, 0 | 0, 0, -a, 0]
        #               [                  ...                          ]
        #               [0,  0,  0,  0, ...,  1, -2,  1, 0 | 0, 0,  0,-a]
        #               [0,  0,  0,  0, ...,  0,  1, -2, 1 | 0, 0,  0,-a]
        #
        #               [-1, 2, -1,  0, ...,  0,  0,  0, 0 | 0, 0, -a, 0]
        #               [0, -1,  2, -1, ...,  0,  0,  0, 0 | 0, 0, -a, 0]
        #               [                  ...                          ]
        #               [0,  0,  0,  0, ..., -1,  2, -1, 0 | 0, 0,  0,-a]
        #               [0,  0,  0,  0, ...,  0, -1,  2,-1 | 0, 0,  0,-a]   (acceleration limit)
        #
        #               [-1,  1,  0, 0, ...,  0,  0,  0, 0 | -v, 0, 0, 0]
        #               [ 0, -1,  1, 0, ...,  0,  0,  0, 0 | -v, 0, 0, 0]
        #               [                  ...                          ]
        #               [0,  0,  0,  0, ...,  0, -1,  1, 0 | 0, -v, 0, 0]
        #               [0,  0,  0,  0, ...,  0,  0, -1, 1 | 0, -v, 0, 0]
        #
        #               [1,  -1,  0, 0, ...,  0,  0,  0, 0 | -v, 0, 0, 0]
        #               [ 0,  1, -1, 0, ...,  0,  0,  0, 0 | -v, 0, 0, 0]
        #               [                  ...                          ]
        #               [0,  0,  0,  0, ...,  0, 1, -1,  0 | 0, -v, 0, 0]
        #               [0,  0,  0,  0, ...,  0, 0,  1, -1 | 0, -v, 0, 0]   (velocity limit)
        #        b_ub = [0, ..., 0] (4*2*(H-2)+4*2*(H-1)) (same as len(A_ub))
        #        A_eq = [1, 0, ..., 0 | 0, ..., 0, 0 | ... | 0, ..., 0, 0 | 0, 0, 0, 0] (4*5, 4*H+4)
        #               [0, ..., 0, 0 | 1, 0, ..., 0 | ... | 0, ..., 0, 0 | 0, 0, 0, 0]
        #               [                           ...                               ]
        #               [0, ..., 0, 0 | 0, ..., 0, 0 | ... | 1, 0, ..., 0 | 0, 0, 0, 0] (p0 condition)
        #               [                           ...                               ]
        #               [0, ..., 0, 1 | 0, ..., 0, 0 | ... | 0, 0, ..., 0 | 0, 0, 0, 0]
        #               [0, ..., 0, 0 | 0, ..., 0, 1 | ... | 0, 0, ..., 0 | 0, 0, 0, 0]
        #               [                           ...                               ]
        #               [0, ..., 0, 0 | 0, ..., 0, 0 | ... | 0, 0, ..., 1 | 0, 0, 0, 0] (pf condition)
        #               [                           ...                               ] (including v0, vf condition)
        #        b_eq = [x0, y0, z0, r0 | xw, yw, zw, rw | xf, yf, zf, rf | 0, ... 0  ] (4*5)

        H = horizon
        n = nb_waypoint

        # objective matrix to minimize time
        v = np.zeros(4*H+4, dtype=np.float)
        v[-4] = n
        v[-3] = H-n
        Q = np.diag(v, k=0)
        c = np.zeros(4*H+4, dtype=np.float)
        c[-2] = n
        c[-1] = H-n

        # Inequality
        # constraint 1: -a_max <= (p(n)-2*p(n-1)+p(n-2))/dt^2 <= a_max
        A1 = (np.diag(np.ones(H), k=0) + np.diag(-2*np.ones(H-1), k=1) + np.diag(np.ones(H-2), k=2))[:-2]
        A2 = (np.diag(-np.ones(H), k=0) + np.diag(np.ones(H-1), k=1))[:-1]
        Az1 = np.zeros_like(A1)
        Az2 = np.zeros_like(A2)
        Aa = np.zeros((4, H-2, 4))  # abs. max acc for each joint
        Av = np.zeros((4, H-1, 4))  # abs. max vel for each joint
        for i in range(4):
            Aa[i, :n, 2] = -max_acc[i]
            Aa[i, n:, 3] = -max_acc[i]
        for i in range(4):
            Av[i, :n, 0] = -max_vel[i]
            Av[i, n:, 1] = -max_vel[i]
        A_ub = np.block([[A1, Az1, Az1, Az1, Aa[0]],
                         [Az1, A1, Az1, Az1, Aa[1]],
                         [Az1, Az1, A1, Az1, Aa[2]],
                         [Az1, Az1, Az1, A1, Aa[3]],

                         [-A1, Az1, Az1, Az1, Aa[0]],
                         [Az1, -A1, Az1, Az1, Aa[1]],
                         [Az1, Az1, -A1, Az1, Aa[2]],
                         [Az1, Az1, Az1, -A1, Aa[3]],

                         [A2, Az2, Az2, Az2, Av[0]],
                         [Az2, A2, Az2, Az2, Av[1]],
                         [Az2, Az2, A2, Az2, Av[2]],
                         [Az2, Az2, Az2, A2, Av[3]],

                         [-A2, Az2, Az2, Az2, Av[0]],
                         [Az2, -A2, Az2, Az2, Av[1]],
                         [Az2, Az2, -A2, Az2, Av[2]],
                         [Az2, Az2, Az2, -A2, Av[3]]])
        b_ub = np.zeros(np.shape(A_ub)[0]).T

        A_eq = []
        b_eq = []
        # Equality
        # constraint 1: p(t0) = p0
        for i in range(4):
            const = np.zeros(4 * H + 4)
            const[i * H + 0] = 1.0
            A_eq.append(const)
            b_eq.append(p0[i])

        # constraint 2: p(tw) = pw (waypoint)
        for i in range(4):
            const = np.zeros(4 * H + 4)
            const[i * H + n - 1] = 1.0
            A_eq.append(const)
            b_eq.append(pw[i])

        # constraint 3: p(tf) = pf
        for i in range(4):
            const = np.zeros(4 * H + 4)
            const[i * H + H - 1] = 1.0
            A_eq.append(const)
            b_eq.append(pf[i])

        # constraint 4: v(t0) = 0
        for i in range(4):
            const = np.zeros(4 * H + 4)
            const[i * H + 0] = 1.0
            const[i * H + 1] = -1.0
            A_eq.append(const)
            b_eq.append(0.0)

        # constraint 5: v(tf) = 0
        for i in range(4):
            const = np.zeros(4 * H + 4)
            const[i * H + H - 2] = 1.0
            const[i * H + H - 1] = -1.0
            A_eq.append(const)
            b_eq.append(0.0)
        A_eq = np.array(A_eq)
        return Q, c, A_ub, b_ub, A_eq, b_eq

    def define_QP_matrix2(self, p0, pw, pf, horizon, nb_waypoint, t_step=0.01):
        # define Q, c, A_eq, b_eq
        # such that
        #    x = [vx0, ..., vx_(H-1) | vy0, ..., vy_(H-1) | vz0, ..., vz_(H-1) | vr0, ..., vr_(H-1) ] (4*H)
        #    Q = [[ 1, -1,  0,  0, ...,  0,  0,  0,  0], (4*H x 4*H)
        #         [-1,  2, -1,  0, ...,  0,  0,  0,  0],
        #         [ 0, -1,  2, -1, ...,  0,  0,  0,  0],
        #         ...,
        #         [ 0,  0,  0,  0, ..., -1,  2, -1,  0],
        #         [ 0,  0,  0,  0, ...,  0, -1,  2, -1],
        #         [ 0,  0,  0,  0, ...,  0,  0, -1,  1]])
        #    c = [0, ..., 0] (4*H)
        # A_eq = [1, 1, ..., 0 | 0, ..., 0, 0 | ... | 0, ..., 0, 0] (4*4, 4*H)
        #        [0, ..., 0, 0 | 1, 1, ..., 0 | ... | 0, ..., 0, 0]
        #        [                           ...                  ]
        #        [0, ..., 0, 0 | 0, ..., 0, 0 | ... | 1, 1, ..., 0] (pw-p0 condition)
        #        [                           ...                  ]
        #        [1, 1, ..., 0 | 0, ..., 0, 0 | ... | 0, ..., 0, 0]
        #        [0, ..., 0, 0 | 1, 1, ..., 0 | ... | 0, ..., 0, 0]
        #        [                           ...                  ]
        #        [0, ..., 0, 0 | 0, ..., 0, 0 | ... | 1, 1, ..., 0] (pf-p0 condition)
        #        [                           ...                  ] (including v0, vf condition)
        # b_eq = [xw-x0, yw-y0, zw-z0, rw-r0 | xf-x0, yf-y0, zf-z0, rf-r0 | 0, ... 0 ] (4*2 + 4*2)

        H = horizon
        n = nb_waypoint

        # objective matrix to minimize acceleration
        v = np.ones(H, dtype=np.float)*2.0
        v[0] = 1.0
        v[-1] = 1.0
        As = np.diag(v, k=0) + np.diag(-1*np.ones(H-1), k=1) + np.diag(-1*np.ones(H-1), k=-1)
        Az = np.zeros((H,H))
        Q = np.block([[As, Az, Az, Az],
                      [Az, As, Az, Az],
                      [Az, Az, As, Az],
                      [Az, Az, Az, As]])
        c = np.zeros(4*H, dtype=np.float)

        # No inequality condition
        A_ub = []
        b_ub = []

        A_eq = []
        b_eq = []
        # Equality
        # constraint 1: p(tw)-p(t0) = pw-p0 (waypoint)
        for i in range(4):
            const = np.zeros(4*H)
            for k in range(n):
                const[i*H + k] = 1.0
            A_eq.append(const)
            b_eq.append((pw[i]-p0[i])/t_step)

        # constraint 2: p(tf)-p(t0) = pf-p0 (end point)
        for i in range(4):
            const = np.zeros(4*H)
            for k in range(H):
                const[i*H + k] = 1.0
            A_eq.append(const)
            b_eq.append((pf[i] - p0[i])/t_step)

        # constraint 3: v(t0) = 0
        for i in range(4):
            const = np.zeros(4*H)
            const[i*H] = 1.0
            A_eq.append(const)
            b_eq.append(0.0)

        # constraint 4: v(tf) = 0
        for i in range(4):
            const = np.zeros(4*H)
            const[i*H + (H-1)] = 1.0
            A_eq.append(const)
            b_eq.append(0.0)
        A_eq = np.array(A_eq)
        return Q, c, A_ub, b_ub, A_eq, b_eq

    def solve_QP(self, Q, c, A_ub, b_ub, A_eq, b_eq):
        """
        Solves a quadratic program
            minimize    (1/2)*x'*Q*x + c'*x
            subject to  A*x = b.
        """
        P = matrix(Q)
        q = matrix(c)
        if len(A_ub) == 0:  G = None
        else:               G = matrix(A_ub)
        if len(b_ub) == 0:  h = None
        else:               h = matrix(b_ub)
        A = matrix(A_eq)
        b = matrix(b_eq)
        solvers.options['show_progress'] = False
        sol = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        return np.array(sol['x']).squeeze()

    def optimize_motion(self, p0, pw, pf, max_vel, max_acc, t_step=0.01, horizon=50, print_out=False, visualize=False):
        # First optimization
        H1 = horizon
        n1 = H1//2
        n2 = H1 - n1

        st1 = time.time()
        Q, c, A_ub, b_ub, A_eq, b_eq = self.define_QP_matrix1(p0, pw, pf, max_vel=max_vel, max_acc=max_acc, horizon=H1, nb_waypoint=n1)
        result1 = self.solve_QP(Q, c, A_ub, b_ub, A_eq, b_eq)
        t_solve1 = time.time() - st1

        # Process the first result
        dt_v1 = result1[-4]
        dt_v2 = result1[-3]
        dt_a1 = np.sqrt(result1[-2])
        dt_a2 = np.sqrt(result1[-1])
        dt1 = max(dt_v1, dt_a1)
        dt2 = max(dt_v2, dt_a2)
        t_total = n1*dt1 + (H1-n1)*dt2
        t1_ratio = (n1)*dt1 / t_total
        t2_ratio = (H1-n1)*dt2 / t_total

        # Calculate time horizon
        t1 = np.arange(n1) * dt1
        t2 = t1[-1] + np.arange(1, n2 + 1) * dt2
        t = np.concatenate((t1, t2), axis=0)

        pos = result1[:-4].reshape(-1, H1).T
        pos_prev = np.insert(pos, 0, pos[0], axis=0)
        pos_prev = np.delete(pos_prev, -1, axis=0)
        dt = np.array([dt1] * (n1) + [dt2] * (H1 - n1))[:, np.newaxis]
        vel = (pos - pos_prev)/dt

        vel_prev = np.insert(vel, 0, vel[0], axis=0)
        vel_prev = np.delete(vel_prev, -1, axis=0)
        dt = np.array([dt1] * (n1) + [dt2] * (H1 - n1))[:, np.newaxis]
        acc = (vel - vel_prev)/dt
        if print_out:
            print("")
            print("< The 1st optimization >")
            print("# of points =", H1)
            print("[dt1, dt2, dt3, dt4] = %3.4f, %3.4f, %3.4f, %3.4f" % (dt_v1, dt_v2, dt_a1, dt_a2))
            print("dt1 (selected) = %3.4f" % dt1)
            print("dt2 (selected)= %3.4f" % dt2)
            print("total time = %3.4f" % t_total)
            print("t1_ratio = %3.4f" % t1_ratio)
            print("t2_ratio = %3.4f" % t2_ratio)
            print("time_solve = %3.4f" % t_solve1)
            print("vel_max = %3.4f, %3.4f, %3.4f, %3.4f" % tuple(np.max(abs(vel), axis=0)))
            print("acc_max = %3.4f, %3.4f, %3.4f, %3.4f" % tuple(np.max(abs(acc), axis=0)))
        if visualize:
            self.plot_joint(t, pos, t[n1-1], pos[n1-1, :])
            self.plot_joint(t, vel, t[n1-1], vel[n1-1, :])
            self.plot_joint(t, acc, t[n1-1], acc[n1-1, :])

        # Second optimization (Use dt1, dt2, calculated from the first result)
        H2 = int(t_total/t_step)+8  # 2 (step/joint), lost from the state change (pos -> vel)
        n1_new = int(H2 * t1_ratio)
        n2_new = H2 - n1_new
        st2 = time.time()
        Q, c, A_ub, b_ub, A_eq, b_eq = self.define_QP_matrix2(p0, pw, pf, horizon=H2, nb_waypoint=n1_new, t_step=t_step)
        result2 = self.solve_QP(Q, c, A_ub, b_ub, A_eq, b_eq)
        t_solve2 = time.time() - st2

        # Process the second result
        vel = result2.reshape(-1,H2).T
        vel_temp = np.insert(vel, 0, vel[0], axis=0)
        pos = p0 + np.cumsum(vel_temp, axis=0)*t_step
        vel_prev = np.insert(vel, 0, vel[0], axis=0)
        vel_prev = np.delete(vel_prev, -1, axis=0)
        acc = (vel - vel_prev)/t_step
        if print_out:
            print("")
            print("< The 2nd optimization >")
            print("# of points = %3.4f" % H2)
            print("n_ratio = %3.4f" % n1_new, ":%3.4f" % n2_new)
            print("time_solve = %3.4f" % t_solve2)
            print("vel_max = %3.4f, %3.4f, %3.4f, %3.4f" % tuple(np.max(abs(vel), axis=0)))
            print("acc_max = %3.4f, %3.4f, %3.4f, %3.4f" % tuple(np.max(abs(acc), axis=0)))

        if visualize:
            t = np.arange(len(pos)) * t_step
            self.plot_joint(t, pos, t[n1_new-1], pos[n1_new-1,:])
            t = np.arange(len(vel)) * t_step
            self.plot_joint(t, vel, t[n1_new-1], vel[n1_new-1,:])
            t = np.arange(len(acc)) * t_step
            self.plot_joint(t, acc, t[n1_new-1], acc[n1_new-1,:])
        return pos, vel, acc, t

    def plot_joint(self, t, p, t_border, p_border):
        # Create plot
        # plt.title('joint angle')
        ax = plt.subplot(411)
        plt.plot(t, p[:, 0], 'b.-', t_border, p_border[0], 'ro-')
        plt.legend(loc='upper center', bbox_to_anchor=(0.9, 2))
        # plt.legend(['q_des', 'q_grey', 'q_red'], ncol=3, bbox_to_anchor=(0.5, 1), loc="lower center")
        # plt.ylim([35, 62])
        ax.set_xticklabels([])
        plt.ylabel('x (mm)')

        ax = plt.subplot(412)
        plt.plot(t, p[:, 1], 'b.-', t_border, p_border[1], 'ro-')
        # plt.ylim([0.14, 0.23])
        ax.set_xticklabels([])
        plt.ylabel('y (mm)')

        ax = plt.subplot(413)
        plt.plot(t, p[:, 2], 'b.-', t_border, p_border[2], 'ro-')
        # plt.ylim([0.14, 0.23])
        ax.set_xticklabels([])
        plt.ylabel('z (mm)')

        plt.subplot(414)
        # plt.plot(t, p[:, 3]*180./np.pi, 'b.-', t_border, p_border[3]*180./np.pi, 'ro-')
        plt.plot(t, p[:, 3], 'b.-', t_border, p_border[3], 'ro-')
        # plt.ylim([-60, 60])
        # plt.legend(['desired', 'actual'])
        plt.ylabel('rot (deg)')
        plt.xlabel('sample number')
        plt.show()