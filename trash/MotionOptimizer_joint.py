import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

class MotionOptimizer:
    def __init__(self):
        pass

    def define_LP_matrix(self, q0, qw, qf, max_acc, horizon, nb_waypoint):
        # define c, A_ub, b_ub, A_eq, b_eq
        # such that x = [q10, ..., q1H | q20, ..., q2H | ... | q60, ..., q6H | dt1**2, dt2**2] (6*H + 2)
        #           c = [0, ..., 0, n, H-n] (6*H + 2)
        #        A_ub = [1, -2,  1,  0, ...,  0,  0,  0, 0 | -a, 0]  (6*2*(H-2) x 6*H+2)
        #               [0,  1, -2,  1, ...,  0,  0,  0, 0 | -a, 0]
        #               [0,  0,  1, -2, ...,  0,  0,  0, 0 | -a, 0]
        #               [0,  0,  0,  1, ...,  0,  0,  0, 0 | -a, 0]
        #               [                  ...                   ]
        #               [0,  0,  0,  0, ..., -2,  1,  0, 0 |  0,-a]
        #               [0,  0,  0,  0, ...,  1, -2,  1, 0 |  0,-a]
        #               [0,  0,  0,  0, ...,  0,  1, -2, 1 |  0,-a]
        #
        #               [-1, 2, -1,  0, ...,  0,  0,  0, 0 | -a, 0]
        #               [0, -1,  2, -1, ...,  0,  0,  0, 0 | -a, 0]
        #               [0,  0, -1,  2, ...,  0,  0,  0, 0 | -a, 0]
        #               [0,  0,  0, -1, ...,  0,  0,  0, 0 | -a, 0]
        #               [                  ...                   ]
        #               [0,  0,  0,  0, ...,  2, -1,  0, 0 |  0,-a]
        #               [0,  0,  0,  0, ..., -1,  2, -1, 0 |  0,-a]
        #               [0,  0,  0,  0, ...,  0, -1,  2,-1 |  0,-a]
        #        b_ub = [0, ..., 0] (6*2*(H-2))
        #        A_eq = [1, 0, ..., 0 | 0, ..., 0, 0 | ... | 0, ..., 0, 0 | 0, 0] (6*5, 6*H+2)
        #               [0, ..., 0, 0 | 1, 0, ..., 0 | ... | 0, ..., 0, 0 | 0, 0]
        #               [                           ...                         ]
        #               [0, ..., 0, 0 | 0, ..., 0, 0 | ... | 1, 0, ..., 0 | 0, 0] (q0 condition)
        #               [                           ...                         ]
        #               [0, ..., 0, 1 | 0, ..., 0, 0 | ... | 0, 0, ..., 0 | 0, 0]
        #               [0, ..., 0, 0 | 0, ..., 0, 1 | ... | 0, 0, ..., 0 | 0, 0]
        #               [                           ...                         ]
        #               [0, ..., 0, 0 | 0, ..., 0, 0 | ... | 0, 0, ..., 1 | 0, 0] (qf condition)
        #               [                           ...                         ]
        #        b_eq = [q10, q20, ..., q60 | q1w, q2w, ..., q6w | q1f, q2f, ..., q6f | 0, ... 0 ] (6*5)

        H = horizon
        n = nb_waypoint

        # objective matrix to minimize time
        c = np.zeros(6*H + 2, dtype=np.float)
        c[-2] = n
        c[-1] = H - n

        # Inequality
        # constraint 1: -a_max <= (q(n)-2*q(n-1)+q(n-2))/dt^2 <= a_max
        A1 = (np.diag(np.ones(H), k=0) + np.diag(-2*np.ones(H-1), k=1) + np.diag(np.ones(H - 2), k=2))[:-2]
        A2 = (np.diag(-np.ones(H), k=0) + np.diag(2*np.ones(H-1), k=1) + np.diag(-np.ones(H - 2), k=2))[:-2]
        Az = np.zeros_like(A1)
        As = np.zeros((6, H - 2, 2))  # abs. max acceleration for each joint
        for i in range(6):
            As[i, :n - 1, 0] = -max_acc[i]
            As[i, n - 1:, 1] = -max_acc[i]

        A_ub = np.block([[A1, Az, Az, Az, Az, Az, As[0]],
                         [Az, A1, Az, Az, Az, Az, As[1]],
                         [Az, Az, A1, Az, Az, Az, As[2]],
                         [Az, Az, Az, A1, Az, Az, As[3]],
                         [Az, Az, Az, Az, A1, Az, As[4]],
                         [Az, Az, Az, Az, Az, A1, As[5]],

                         [A2, Az, Az, Az, Az, Az, As[0]],
                         [Az, A2, Az, Az, Az, Az, As[1]],
                         [Az, Az, A2, Az, Az, Az, As[2]],
                         [Az, Az, Az, A2, Az, Az, As[3]],
                         [Az, Az, Az, Az, A2, Az, As[4]],
                         [Az, Az, Az, Az, Az, A2, As[5]]])
        b_ub = np.zeros(np.shape(A_ub)[0]).T

        # Equality
        A_eq = []
        b_eq = []

        # constraint 1: q(t0) = q0
        for i in range(6):
            const = np.zeros(6 * H + 2)
            const[i * H + 0] = 1.0
            A_eq.append(const)
            b_eq.append(q0[i])

        # constraint 2: q(tw) = qw (waypoint)
        for i in range(6):
            const = np.zeros(6 * H + 2)
            const[i * H + n - 1] = 1.0
            A_eq.append(const)
            b_eq.append(qw[i])

        # constraint 3: q(tf) = qf
        for i in range(6):
            const = np.zeros(6 * H + 2)
            const[i * H + H - 1] = 1.0
            A_eq.append(const)
            b_eq.append(qf[i])

        # constraint 4: v(t0) = 0
        for i in range(6):
            const = np.zeros(6 * H + 2)
            const[i * H + 0] = 1.0
            const[i * H + 1] = -1.0
            A_eq.append(const)
            b_eq.append(0.0)

        # constraint 5: v(tf) = 0
        for i in range(6):
            const = np.zeros(6 * H + 2)
            const[i * H + H - 2] = 1.0
            const[i * H + H - 1] = -1.0
            A_eq.append(const)
            b_eq.append(0.0)
        A_eq = np.array(A_eq)
        return c, A_ub, b_ub, A_eq, b_eq

    def define_QP_matrix(self, q0, qw, qf, horizon, nb_waypoint):
        # define Q, c, A_eq, b_eq
        # such that
        #    x = [q10, ..., q1H | ... | q60, ..., q6H | a1*dt1**2, ..., a6*dt1**2 | a1*dt2**2, ... a6*dt2**2] (6*H+12)
        #    Q = [[0., 0., 0., ..., 0., 0., 0.], (6*H+12 x 6*H+12)
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         ...,
        #         [0., 0., 0., ..., 1., 0., 0.],
        #         [0., 0., 0., ..., 0., 1., 0.],
        #         [0., 0., 0., ..., 0., 0., 1.]])
        #    c = [0, ..., 0] (6*H+12)
        # A_eq = [[1., -2., 1., ..., 0., 0., 0.], (6*(H-2)+6*5, 6*H+12)
        #         [0., 1., -2., ..., 0., 0., 0.],
        #         [0., 0., 1., ..., 0., 0., 0.],
        #         ...,
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.]]) (including q0, qw, qf, v0, vf condition)
        # b_eq = [0, ..., 0 | q10, q20, ..., q60 | q1w, q2w, ..., q6w | q1f, q2f, ..., q6f | 0, ... 0 ] (6*(H-2)+6*5)

        H = horizon

        # objective matrix to minimize smoothness
        v1 = np.zeros(6 * H, dtype=np.float)
        v2 = np.ones(12, dtype=np.float)
        v = np.concatenate((v1, v2))
        Q = np.diag(v, k=0)
        c = np.zeros(6 * H + 12, dtype=np.float)

        # Equality
        # constraint 1: q(n)-2*q(n-1)+q(n-2) - a*dt^2 = 0
        A1 = (np.diag(np.ones(H), k=0) + np.diag(-2 * np.ones(H - 1), k=1) + np.diag(np.ones(H - 2), k=2))[:-2]
        Az = np.zeros_like(A1)
        As = np.zeros((6, H - 2, 12))  # abs. max acceleration for each joint
        for i in range(6):
            As[i, :H // 2 - 1, i] = -1
            As[i, H // 2 - 1:, i + 6] = -1
        A_eq = np.block([[A1, Az, Az, Az, Az, Az, As[0]],
                         [Az, A1, Az, Az, Az, Az, As[1]],
                         [Az, Az, A1, Az, Az, Az, As[2]],
                         [Az, Az, Az, A1, Az, Az, As[3]],
                         [Az, Az, Az, Az, A1, Az, As[4]],
                         [Az, Az, Az, Az, Az, A1, As[5]]])
        b_eq = np.zeros(np.shape(A_eq)[0])

        # constraint 1: q(t0) = q0
        for i in range(6):
            const = np.zeros(6 * H + 12)
            const[i * H + 0] = 1.0
            A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
            b_eq = np.insert(b_eq, len(b_eq), q0[i])

        # constraint 2: q(tw) = qw (waypoint)
        for i in range(6):
            const = np.zeros(6 * H + 12)
            const[i*H + nb_waypoint-1] = 1.0
            A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
            b_eq = np.insert(b_eq, len(b_eq), qw[i])

        # constraint 3: q(tf) = qf
        for i in range(6):
            const = np.zeros(6 * H + 12)
            const[i * H + H - 1] = 1.0
            A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
            b_eq = np.insert(b_eq, len(b_eq), qf[i])

        # constraint 4: v(t0) = 0
        for i in range(6):
            const = np.zeros(6 * H + 12)
            const[i * H + 0] = 1.0
            const[i * H + 1] = -1.0
            A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
            b_eq = np.insert(b_eq, len(b_eq), 0.0)

        # constraint 5: v(tf) = 0
        for i in range(6):
            const = np.zeros(6 * H + 12)
            const[i * H + H - 2] = 1.0
            const[i * H + H - 1] = -1.0
            A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
            b_eq = np.insert(b_eq, len(b_eq), 0.0)
        return Q, c, A_eq, b_eq

    def solve_LP(self, c, A_ub, b_ub, A_eq, b_eq):
        # arg min cT*x
        # such that A_ub*x <= b_ub
        #           A_eq*x = b_eq
        result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))
        return result

    def solve_QP(self, Q, c, A_eq, b_eq):
        # arg min (1/2)*xT*Q*x + cT*x
        # such that A_eq*x = b_eq
        #           x >= 0
        A = np.block([[Q, A_eq.T], [A_eq, np.eye(np.shape(A_eq)[0])]])
        b = np.concatenate((-c, b_eq))
        result = np.linalg.lstsq(A, b)
        return result[0][:np.shape(Q)[0]]

    def optimize_motion(self, q0, qw, qf, max_acc, horizon1, horizon2, visualize=False):
        # First optimization
        H1 = horizon1
        n1 = H1//2
        c, A_ub, b_ub, A_eq, b_eq = self.define_LP_matrix(q0, qw, qf, max_acc=max_acc, horizon=H1, nb_waypoint=n1)
        result1 = self.solve_LP(c, A_ub, b_ub, A_eq, b_eq)

        # Process the first result
        dt1 = np.sqrt(result1.x[-2])
        dt2 = np.sqrt(result1.x[-1])
        print (dt1, dt2)
        # x = result.x[:-2].reshape(6, -1).T    # not used
        # print("dt1=", dt1)
        # print("dt2=", dt2)
        total = n1*dt1 + (H1-n1)*dt2
        t1_ratio = (n1)*dt1 / total
        t2_ratio = (H1-n1)*dt2 / total
        print("[Result of the 1st optimization]")
        # print("total time =", total)
        print("t1_ratio =", t1_ratio)
        print("t2_ratio =", t2_ratio)
        n1 = int(horizon2*t1_ratio)
        n2 = horizon2-n1
        print("n_ratio =", n1, ":", n2)

        # Second optimization
        H2 = horizon2
        Q, c, A_eq, b_eq = self.define_QP_matrix(q0, qw, qf, horizon=H2, nb_waypoint=n1)
        result2 = self.solve_QP(Q, c, A_eq, b_eq)

        # Process the second result
        #    x = [q10, ..., q1H | ... | q60, ..., q6H | a1*dt1**2, ..., a6*dt1**2 | a1*dt2**2, ... a6*dt2**2] (6*H+12)
        qs = result2[:6*H2].reshape(-1,H2).T
        adt1sq = result2[-12:-6]
        adt2sq = result2[-6:]
        a = np.array(max_acc)
        dt1 = max(np.sqrt(abs(adt1sq/a)))
        dt2 = max(np.sqrt(abs(adt2sq/a)))
        print("[Result of the 2nd optimization]")
        # print("qs =", qs)
        print("dt1 =", dt1)
        print("dt2 =", dt2)

        # time horizon
        t1 = np.arange(n1)*dt1
        t2 = t1[-1] + np.arange(1, n2+1)*dt2
        time = np.concatenate((t1, t2), axis=0)

        if visualize:
            self.plot_joint(time, qs, time[n1], qs[n1,:])
        return qs, time

    def plot_joint(self, t, qs, t_border, qs_border):
        # Create plot
        # plt.title('joint angle')
        ax = plt.subplot(611)
        plt.plot(t, qs[:, 0] * 180. / np.pi, 'b.-', t_border, qs_border[0] * 180. / np.pi, 'ro-')
        plt.legend(loc='upper center', bbox_to_anchor=(0.9, 2))
        # plt.legend(['q_des', 'q_grey', 'q_red'], ncol=3, bbox_to_anchor=(0.5, 1), loc="lower center")
        # plt.ylim([35, 62])
        ax.set_xticklabels([])
        plt.ylabel('q0 ($^\circ$)')

        ax = plt.subplot(612)
        plt.plot(t, qs[:, 1] * 180. / np.pi, 'b.-', t_border, qs_border[1] * 180. / np.pi, 'ro-')
        # plt.ylim([-10, 12])
        ax.set_xticklabels([])
        plt.ylabel('q2 ($^\circ$)')

        ax = plt.subplot(613)
        plt.plot(t, qs[:, 2], 'b.-', t_border, qs_border[2], 'ro-')
        # plt.ylim([0.14, 0.23])
        ax.set_xticklabels([])
        plt.ylabel('q3 (m)')

        ax = plt.subplot(614)
        plt.plot(t, qs[:, 3] * 180. / np.pi, 'b.-', t_border, qs_border[3] * 180. / np.pi, 'ro-')
        # plt.ylim([-90, 70])
        ax.set_xticklabels([])
        plt.ylabel('q4 ($^\circ$)')

        ax = plt.subplot(615)
        plt.plot(t, qs[:, 4] * 180. / np.pi, 'b.-', t_border, qs_border[4] * 180. / np.pi, 'ro-')
        # plt.ylim([-60, 60])
        ax.set_xticklabels([])
        plt.ylabel('q5 ($^\circ$)')

        plt.subplot(616)
        plt.plot(t, qs[:, 5] * 180. / np.pi, 'b.-', t_border, qs_border[5] * 180. / np.pi, 'ro-')
        # plt.ylim([-60, 60])
        # plt.legend(['desired', 'actual'])
        plt.ylabel('q6 ($^\circ$)')
        plt.xlabel('sample number')
        plt.show()