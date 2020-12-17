import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import time


class PegMotionOptimizer_2wp:   # two way points
    def __init__(self):
        pass

    def define_QP_matrix1(self, q0, qw1, dqw1, qw2, dqw2, qf, max_vel, max_acc, horizon, nb_waypoints):
        # define Q, c, A_ub, b_ub, A_eq, b_eq
        # such that x = [q10, ..., q1(H-1) | q20, ..., q2(H-1) | ... | q60, ..., q6(H-1) | dt1, dt2, dt3**2, dt4**2] (6*H + 4)
        #          Q = [[ 0,  0,  0,  0, ...,  0,  0,  0,  0], (6*H+4) x (6*H+4)
        #               [ 0,  0,  0,  0, ...,  0,  0,  0,  0],
        #               [                ...,               ],
        #               [ 0,  0,  0,  0, ...,  n,  0,  0,  0],
        #               [ 0,  0,  0,  0, ...,  0,(H-n),0,  0],
        #               [ 0,  0,  0,  0, ...,  0,  0,  0,  0],
        #               [ 0,  0,  0,  0, ...,  0,  0,  0,  0]])
        #           c = [0, ..., 0, n, H-n] (6*H + 4)
        #        A_ub = [1, -2,  1,  0, ...,  0,  0,  0, 0 | 0, 0, -a, 0]  (6*2*(H-2)+6*2*(H-1)) x (6*H+4)
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
        #        b_ub = [0, ..., 0] (6*2*(H-2)+6*2*(H-1)) (same as len(A_ub))
        #        A_eq = [1, 0, ..., 0 | 0, ..., 0, 0 | ... | 0, ..., 0, 0 | 0, 0, 0, 0] (6*5+.., 6*H+4)
        #               [0, ..., 0, 0 | 1, 0, ..., 0 | ... | 0, ..., 0, 0 | 0, 0, 0, 0]
        #               [                           ...                               ]
        #               [0, ..., 0, 0 | 0, ..., 0, 0 | ... | 1, 0, ..., 0 | 0, 0, 0, 0] (p0 condition)
        #               [                           ...                               ]
        #               [0, ..., 0, 1 | 0, ..., 0, 0 | ... | 0, 0, ..., 0 | 0, 0, 0, 0]
        #               [0, ..., 0, 0 | 0, ..., 0, 1 | ... | 0, 0, ..., 0 | 0, 0, 0, 0]
        #               [                           ...                               ]
        #               [0, ..., 0, 0 | 0, ..., 0, 0 | ... | 0, 0, ..., 1 | 0, 0, 0, 0] (pf condition)
        #               [                           ...                               ] (including v0, vf condition)
        #               [                            ...                              ] (velocity ratio at waypoint)
        #        b_eq = [q10, ..., q60 | q1w, ..., q6w | q1f, ..., q6f | v10, ..., v60 | v1f, ...,  v6f ] (6*5)

        H = horizon
        n1, n2 = nb_waypoints

        # objective matrix to minimize time
        v = np.zeros(6*H+6, dtype=np.float)
        v[-6] = n1
        v[-5] = n2-n1
        v[-4] = H-n2
        Q = np.diag(v, k=0)
        c = np.zeros(6*H+6, dtype=np.float)
        c[-3] = n1
        c[-2] = n2-n1
        c[-1] = H-n2

        # Inequality
        # constraint 1: -a_max <= (q(n)-2*q(n-1)+q(n-2))/dt^2 <= a_max
        A1 = (np.diag(np.ones(H), k=0) + np.diag(-2*np.ones(H-1), k=1) + np.diag(np.ones(H-2), k=2))[:-2]
        A2 = (np.diag(-np.ones(H), k=0) + np.diag(np.ones(H-1), k=1))[:-1]
        Az1 = np.zeros_like(A1)
        Az2 = np.zeros_like(A2)
        Aa = np.zeros((6, H-2, 6))  # abs. max acc for each joint
        Av = np.zeros((6, H-1, 6))  # abs. max vel for each joint
        for i in range(6):
            Aa[i, :n1, 3] = -max_acc[i]
            Aa[i, n1:n2, 4] = -max_acc[i]
            Aa[i, n2:, 5] = -max_acc[i]
        for i in range(6):
            Av[i, :n1, 0] = -max_vel[i]
            Av[i, n1:n2, 1] = -max_vel[i]
            Av[i, n2:, 2] = -max_vel[i]
        A_ub = np.block([[A1, Az1, Az1, Az1, Az1, Az1, Aa[0]],
                         [Az1, A1, Az1, Az1, Az1, Az1, Aa[1]],
                         [Az1, Az1, A1, Az1, Az1, Az1, Aa[2]],
                         [Az1, Az1, Az1, A1, Az1, Az1, Aa[3]],
                         [Az1, Az1, Az1, Az1, A1, Az1, Aa[4]],
                         [Az1, Az1, Az1, Az1, Az1, A1, Aa[5]],

                         [-A1, Az1, Az1, Az1, Az1, Az1, Aa[0]],
                         [Az1, -A1, Az1, Az1, Az1, Az1, Aa[1]],
                         [Az1, Az1, -A1, Az1, Az1, Az1, Aa[2]],
                         [Az1, Az1, Az1, -A1, Az1, Az1, Aa[3]],
                         [Az1, Az1, Az1, Az1, -A1, Az1, Aa[4]],
                         [Az1, Az1, Az1, Az1, Az1, -A1, Aa[5]],

                         [A2, Az2, Az2, Az2, Az2, Az2, Av[0]],
                         [Az2, A2, Az2, Az2, Az2, Az2, Av[1]],
                         [Az2, Az2, A2, Az2, Az2, Az2, Av[2]],
                         [Az2, Az2, Az2, A2, Az2, Az2, Av[3]],
                         [Az2, Az2, Az2, Az2, A2, Az2, Av[4]],
                         [Az2, Az2, Az2, Az2, Az2, A2, Av[5]],

                         [-A2, Az2, Az2, Az2, Az2, Az2, Av[0]],
                         [Az2, -A2, Az2, Az2, Az2, Az2, Av[1]],
                         [Az2, Az2, -A2, Az2, Az2, Az2, Av[2]],
                         [Az2, Az2, Az2, -A2, Az2, Az2, Av[3]],
                         [Az2, Az2, Az2, Az2, -A2, Az2, Av[4]],
                         [Az2, Az2, Az2, Az2, Az2, -A2, Av[5]]])
        b_ub = np.zeros(np.shape(A_ub)[0]).T

        A_eq = []
        b_eq = []
        # Equality
        # constraint 1: q(t0) = q0
        for i in range(6):
            const = np.zeros(6*H + 6)
            const[i * H + 0] = 1.0
            A_eq.append(const)
            b_eq.append(q0[i])

        # constraint 2: q(tw1) = qw1 (waypoint)
        for i in range(6):
            const = np.zeros(6*H + 6)
            const[i * H + n1 - 1] = 1.0
            A_eq.append(const)
            b_eq.append(qw1[i])

        # constraint 3: v(tw1) = dqw1*alpha (velocity ratio at waypoint)
        for i in range(5):
            const = np.zeros(6*H + 6)
            const[i*H+n1-1] = -dqw1[i+1]
            const[i*H+n1] = dqw1[i+1]
            const[(i+1)*H+n1-1] = dqw1[i]
            const[(i+1)*H+n1] = -dqw1[i]
            A_eq.append(const)
            b_eq.append(0.0)

        # constraint 4: q(tw2) = qw2 (waypoint)
        for i in range(6):
            const = np.zeros(6*H + 6)
            const[i * H + n2 - 1] = 1.0
            A_eq.append(const)
            b_eq.append(qw2[i])

        # constraint 5: v(tw2) = dqw2*alpha (velocity ratio at waypoint)
        for i in range(5):
            const = np.zeros(6*H + 6)
            const[i*H+n2-1] = -dqw2[i+1]
            const[i*H+n2] = dqw2[i+1]
            const[(i+1)*H+n2-1] = dqw2[i]
            const[(i+1)*H+n2] = -dqw2[i]
            A_eq.append(const)
            b_eq.append(0.0)

        # constraint 6: q(tf) = qf
        for i in range(6):
            const = np.zeros(6*H + 6)
            const[i * H + H - 1] = 1.0
            A_eq.append(const)
            b_eq.append(qf[i])

        # constraint 7: v(t0) = 0
        for i in range(6):
            const = np.zeros(6*H + 6)
            const[i * H + 0] = 1.0
            const[i * H + 1] = -1.0
            A_eq.append(const)
            b_eq.append(0.0)

        # constraint 8: v(tf) = 0
        for i in range(6):
            const = np.zeros(6*H + 6)
            const[i * H + H - 2] = 1.0
            const[i * H + H - 1] = -1.0
            A_eq.append(const)
            b_eq.append(0.0)
        A_eq = np.array(A_eq)
        return Q, c, A_ub, b_ub, A_eq, b_eq

    def define_QP_matrix2(self, q0, qw1, dqw1, qw2, dqw2, qf, horizon, nb_waypoint, t_step=0.01):
        # define Q, c, A_eq, b_eq
        # such that
        #    x = [v10, ..., v1(H-1) | v20, ..., v2(H-1) | ... | v60, ..., v6(H-1) ] (6*H)
        #    Q = [[ 1, -1,  0,  0, ...,  0,  0,  0,  0], (6*H x 6*H)
        #         [-1,  2, -1,  0, ...,  0,  0,  0,  0],
        #         [ 0, -1,  2, -1, ...,  0,  0,  0,  0],
        #         ...,
        #         [ 0,  0,  0,  0, ..., -1,  2, -1,  0],
        #         [ 0,  0,  0,  0, ...,  0, -1,  2, -1],
        #         [ 0,  0,  0,  0, ...,  0,  0, -1,  1]])
        #    c = [0, ..., 0] (6*H)
        # A_eq = [1, 1, ..., 0 | 0, ..., 0, 0 | ... | 0, ..., 0, 0] (6*4+.., 6*H)
        #        [0, ..., 0, 0 | 1, 1, ..., 0 | ... | 0, ..., 0, 0]
        #        [                           ...                  ]
        #        [0, ..., 0, 0 | 0, ..., 0, 0 | ... | 1, 1, ..., 0] (qw-q0 condition)
        #        [                           ...                  ]
        #        [1, 1, ..., 0 | 0, ..., 0, 0 | ... | 0, ..., 0, 0]
        #        [0, ..., 0, 0 | 1, 1, ..., 0 | ... | 0, ..., 0, 0]
        #        [                           ...                  ]
        #        [0, ..., 0, 0 | 0, ..., 0, 0 | ... | 1, 1, ..., 0] (qf-q0 condition)
        #        [                           ...                  ] (including v0, vf condition)
        #        [                           ...                  ] (including velocity ratio at waypoint)
        # b_eq = [q1w-q10, q2w-q20, .., q6w-q60 | q1f-q10, q2f-q20, ..., q6f-q60 | 0, ... 0 ] (6*4)

        H = horizon
        n1,n2 = nb_waypoint

        # objective matrix to minimize acceleration
        v = np.ones(H, dtype=np.float)*2.0
        v[0] = 1.0
        v[-1] = 1.0
        As = np.diag(v, k=0) + np.diag(-1*np.ones(H-1), k=1) + np.diag(-1*np.ones(H-1), k=-1)
        Az = np.zeros((H,H))
        Q = np.block([[As, Az, Az, Az, Az, Az],
                      [Az, As, Az, Az, Az, Az],
                      [Az, Az, As, Az, Az, Az],
                      [Az, Az, Az, As, Az, Az],
                      [Az, Az, Az, Az, As, Az],
                      [Az, Az, Az, Az, Az, As]])
        c = np.zeros(6*H, dtype=np.float)

        # No inequality condition
        A_ub = []
        b_ub = []

        # Equality
        A_eq = []
        b_eq = []
        # constraint 1: q(tw1)-q(t0) = qw1-q0 (waypoint)
        for i in range(6):
            const = np.zeros(6*H)
            for k in range(n1):
                const[i*H + k] = 1.0
            A_eq.append(const)
            b_eq.append((qw1[i]-q0[i])/t_step)

        # constraint 2: q(tw2)-q(t0) = qw2-q0 (waypoint2)
        for i in range(6):
            const = np.zeros(6 * H)
            for k in range(n2):
                const[i * H + k] = 1.0
            A_eq.append(const)
            b_eq.append((qw2[i] - q0[i]) / t_step)

        # constraint 3: q(tf)-q(t0) = qf-q0 (end point)
        for i in range(6):
            const = np.zeros(6*H)
            for k in range(H):
                const[i*H + k] = 1.0
            A_eq.append(const)
            b_eq.append((qf[i] - q0[i])/t_step)

        # constraint 4: v(t0) = 0
        for i in range(6):
            const = np.zeros(6*H)
            const[i*H] = 1.0
            A_eq.append(const)
            b_eq.append(0.0)

        # constraint 5: v(tf) = 0
        for i in range(6):
            const = np.zeros(6*H)
            const[i*H + (H-1)] = 1.0
            A_eq.append(const)
            b_eq.append(0.0)

        # constraint 6: v(tw1) = dvw1*alpha (velocity ratio at waypoint)
        for i in range(5):
            const = np.zeros(6*H)
            const[i*H+n1] = dqw1[i+1]
            const[(i+1)*H+n1] = -dqw1[i]
            A_eq.append(const)
            b_eq.append(0.0)

        # constraint 7: v(tw2) = dvw2*alpha (velocity ratio at waypoint)
        for i in range(5):
            const = np.zeros(6*H)
            const[i*H+n2] = dqw2[i+1]
            const[(i+1)*H+n2] = -dqw2[i]
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

    def optimize_motion(self, q0, qw1, dqw1, qw2, dqw2, qf, max_vel, max_acc, t_step=0.01, horizon=50, print_out=False, visualize=False):
        # First optimization
        H1 = horizon
        n11 = H1//3
        n12 = H1//3*2

        st1 = time.time()
        Q, c, A_ub, b_ub, A_eq, b_eq = self.define_QP_matrix1(q0, qw1, dqw1, qw2, dqw2, qf, max_vel=max_vel, max_acc=max_acc, horizon=H1, nb_waypoints=[n11, n12])
        result1 = self.solve_QP(Q, c, A_ub, b_ub, A_eq, b_eq)
        t_solve1 = time.time() - st1

        # Process the first result
        dt_v1 = result1[-6]
        dt_v2 = result1[-5]
        dt_v3 = result1[-4]
        dt_a1 = np.sqrt(result1[-3])
        dt_a2 = np.sqrt(result1[-2])
        dt_a3 = np.sqrt(result1[-1])
        dt1 = max(dt_v1, dt_a1)
        dt2 = max(dt_v2, dt_a2)
        dt3 = max(dt_v3, dt_a3)
        t_total = n11*dt1 + (n12-n11)*dt2 + (H1-n12)*dt3
        t1_ratio = (n11)*dt1 / t_total
        t2_ratio = (n12-n11)*dt2 / t_total
        t3_ratio = (H1-n12)*dt3 / t_total

        # Calculate time horizon
        t1 = np.arange(n11) * dt1
        t2 = t1[-1] + np.arange(1, n12-n11+1) * dt2
        t3 = t2[-1] + np.arange(1, H1-n12+1) * dt3
        t = np.concatenate((t1, t2, t3), axis=0)

        pos = result1[:-6].reshape(-1, H1).T
        pos_prev = np.insert(pos, 0, pos[0], axis=0)
        pos_prev = np.delete(pos_prev, -1, axis=0)
        dt = np.array([dt1]*(n11) + [dt2]*(n12-n11) + [dt3]*(H1-n12))[:, np.newaxis]
        vel = (pos - pos_prev)/dt

        vel_prev = np.insert(vel, 0, vel[0], axis=0)
        vel_prev = np.delete(vel_prev, -1, axis=0)
        dt = np.array([dt1]*(n11) + [dt2]*(n12-n11) + [dt3]*(H1-n12))[:, np.newaxis]
        acc = (vel - vel_prev)/dt
        if print_out:
            print("")
            print("< The 1st optimization >")
            print("# of points =", H1)
            print("[dt1, dt2, dt3, dt4, dt5, dt6] = %3.4f, %3.4f, %3.4f, %3.4f, %3.4f, %3.4f" % (dt_v1, dt_v2, dt_v3, dt_a1, dt_a2, dt_a3))
            print("dt1 (selected) = %3.4f" % dt1)
            print("dt2 (selected)= %3.4f" % dt2)
            print("dt3 (selected)= %3.4f" % dt3)
            print("total time = %3.4f" % t_total)
            print("t1_ratio = %3.4f" % t1_ratio)
            print("t2_ratio = %3.4f" % t2_ratio)
            print("t3_ratio = %3.4f" % t3_ratio)
            print("time_solve = %3.4f" % t_solve1)
            print("vel_max = %3.4f, %3.4f, %3.4f, %3.4f, %3.4f, %3.4f" % tuple(np.max(abs(vel), axis=0)))
            print("acc_max = %3.4f, %3.4f, %3.4f, %3.4f, %3.4f, %3.4f" % tuple(np.max(abs(acc), axis=0)))
        if visualize:
            self.plot_joint(t, pos, t[n11-1], pos[n11-1,:], t[n12-1], pos[n12-1,:], nb_figure=1, hold=False)
            self.plot_joint(t, vel, t[n11-1], vel[n11-1,:], t[n12-1], vel[n12-1,:], nb_figure=2, hold=False)
            self.plot_joint(t, acc, t[n11-1], acc[n11-1,:], t[n12-1], acc[n12-1,:], nb_figure=3, hold=True)

        # Second optimization (Use dt1, dt2, calculated from the first result)
        H2 = int(t_total/t_step)
        n1_new = int(H2 * t1_ratio)
        n2_new = int(H2 * (t1_ratio+t2_ratio))
        st2 = time.time()
        Q, c, A_ub, b_ub, A_eq, b_eq = self.define_QP_matrix2(q0, qw1, dqw1, qw2, dqw2, qf, horizon=H2, nb_waypoint=[n1_new, n2_new], t_step=t_step)
        result2 = self.solve_QP(Q, c, A_ub, b_ub, A_eq, b_eq)
        t_solve2 = time.time() - st2

        # Process the second result
        vel = result2.reshape(-1,H2).T
        vel_temp = np.insert(vel, 0, vel[0], axis=0)
        pos = q0 + np.cumsum(vel_temp, axis=0)*t_step
        vel_prev = np.insert(vel, 0, vel[0], axis=0)
        vel_prev = np.delete(vel_prev, -1, axis=0)
        acc = (vel - vel_prev)/t_step
        if print_out:
            print("")
            print("< The 2nd optimization >")
            print("# of points = %d" % H2)
            print("n_ratio = %d" % n1_new, ":%d" % (n2_new-n1_new), ":%d" % (H2-n2_new))
            print("time_solve = %3.4f" % t_solve2)
            print("vel_max = %3.4f, %3.4f, %3.4f, %3.4f, %3.4f, %3.4f" % tuple(np.max(abs(vel), axis=0)))
            print("acc_max = %3.4f, %3.4f, %3.4f, %3.4f, %3.4f, %3.4f" % tuple(np.max(abs(acc), axis=0)))

        if visualize:
            t = np.arange(len(pos)) * t_step
            self.plot_joint(t, pos, t[n1_new-1], pos[n1_new-1,:], t[n2_new-1], pos[n2_new-1, :], nb_figure=1, hold=False)
            t = np.arange(len(vel)) * t_step
            self.plot_joint(t, vel, t[n1_new-1], vel[n1_new-1,:], t[n2_new-1], vel[n2_new-1, :], nb_figure=2, hold=False)
            t = np.arange(len(acc)) * t_step
            self.plot_joint(t, acc, t[n1_new-1], acc[n1_new-1,:], t[n2_new-1], acc[n2_new-1, :], nb_figure=3, hold=True)
        t = np.arange(len(pos)) * t_step
        return pos, vel, acc, t

    def plot_joint(self, t, q, t_border, q_border, t_border2, q_border2, nb_figure, hold):
        # Create plot
        # plt.title('joint angle')
        plt.figure(nb_figure)
        ax = plt.subplot(611)
        plt.plot(t, q[:, 0], 'b.-', t_border, q_border[0], 'ro-', t_border2, q_border2[0], 'ro-')
        plt.legend(loc='upper center', bbox_to_anchor=(0.9, 2))
        # plt.legend(['q_des', 'q_grey', 'q_red'], ncol=3, bbox_to_anchor=(0.5, 1), loc="lower center")
        # plt.ylim([35, 62])
        ax.set_xticklabels([])
        plt.ylabel('q1 (rad)')

        ax = plt.subplot(612)
        plt.plot(t, q[:, 1], 'b.-', t_border, q_border[1], 'ro-', t_border2, q_border2[1], 'ro-')
        # plt.ylim([0.14, 0.23])
        ax.set_xticklabels([])
        plt.ylabel('q2 (rad)')

        ax = plt.subplot(613)
        plt.plot(t, q[:, 2], 'b.-', t_border, q_border[2], 'ro-', t_border2, q_border2[2], 'ro-')
        # plt.ylim([0.14, 0.23])
        ax.set_xticklabels([])
        plt.ylabel('q3 (mm)')

        ax = plt.subplot(614)
        plt.plot(t, q[:, 3], 'b.-', t_border, q_border[3], 'ro-', t_border2, q_border2[3], 'ro-')
        # plt.ylim([0.14, 0.23])
        ax.set_xticklabels([])
        plt.ylabel('q4 (rad)')

        ax = plt.subplot(615)
        plt.plot(t, q[:, 4], 'b.-', t_border, q_border[4], 'ro-', t_border2, q_border2[4], 'ro-')
        # plt.ylim([0.14, 0.23])
        ax.set_xticklabels([])
        plt.ylabel('q5 (rad)')

        plt.subplot(616)
        plt.plot(t, q[:, 5], 'b.-', t_border, q_border[5], 'ro-', t_border2, q_border2[5], 'ro-')
        # plt.ylim([-60, 60])
        # plt.legend(['desired', 'actual'])
        plt.ylabel('q6 (rad)')
        plt.xlabel('time (s)')
        if hold:
            plt.show()
