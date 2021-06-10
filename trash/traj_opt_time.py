import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def plot_joint(t, q_des, q_act):
    # Create plot
    # plt.title('joint angle')
    ax = plt.subplot(611)
    plt.plot(t, q_des[:,0]*180./np.pi, 'b-')
    plt.plot(t, q_act[:, 0] * 180. / np.pi, 'r-')
    plt.legend(loc='upper center', bbox_to_anchor=(0.9,2))
    # plt.legend(['q_des', 'q_grey', 'q_red'], ncol=3, bbox_to_anchor=(0.5, 1), loc="lower center")
    # plt.ylim([35, 62])
    ax.set_xticklabels([])
    plt.ylabel('q0 ($^\circ$)')

    ax = plt.subplot(612)
    plt.plot(t, q_des[:,1]*180./np.pi, 'b-', t, q_act[:, 1] * 180. / np.pi, 'r-')
    # plt.ylim([-10, 12])
    ax.set_xticklabels([])
    plt.ylabel('q2 ($^\circ$)')

    ax = plt.subplot(613)
    plt.plot(t, q_des[:, 2], 'b-', t, q_act[:, 2], 'r-')
    # plt.ylim([0.14, 0.23])
    ax.set_xticklabels([])
    plt.ylabel('q3 (m)')

    ax = plt.subplot(614)
    plt.plot(t, q_des[:, 3]*180./np.pi, 'b-', t, q_act[:, 3]*180./np.pi, 'r-')
    # plt.ylim([-90, 70])
    ax.set_xticklabels([])
    plt.ylabel('q4 ($^\circ$)')

    ax = plt.subplot(615)
    plt.plot(t, q_des[:, 4]*180./np.pi, 'b-', t, q_act[:, 4]*180./np.pi, 'r-')
    # plt.ylim([-60, 60])
    ax.set_xticklabels([])
    plt.ylabel('q5 ($^\circ$)')

    plt.subplot(616)
    plt.plot(t, q_des[:, 5]*180./np.pi, 'b-', t, q_act[:, 5]*180./np.pi, 'r-')
    # plt.ylim([-60, 60])
    # plt.legend(['desired', 'actual'])
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('sample number')
    plt.show()


def define_matrix(q0, qw, qf, horizon, abs_max_acc):
    H = horizon
    a = abs_max_acc

    # objective matrix to minimize time
    c = np.zeros(H+2, dtype=np.float)
    c[-1] = 1
    c[-2] = 1

    # Inequality
    # constraint 1: -a_max <= (q(n)-2*q(n-1)+q(n-2))/dt^2 <= a_max
    A1 = (np.diag(np.ones(H+2), k=0) + np.diag(-2*np.ones(H+1), k=1) + np.diag(np.ones(H), k=2))[:-4]
    A1[:H//2-1, 2*H//2] = -a
    A1[H//2-1:, 2*H//2+1] = -a
    A2 = (np.diag(-np.ones(H+2), k=0) + np.diag(2*np.ones(H+1), k=1) + np.diag(-np.ones(H), k=2))[:-4]
    A2[:H//2-1, H] = -a
    A2[H//2-1:, H+1] = -a
    A_ub = np.block([[A1],[A2]])
    b_ub = np.zeros(2*(2*(H//2-2)+2)).T

    # Equality
    # constraint 1: q(t0) = q0
    const = np.zeros(H + 2)
    const[0] = 1.0
    A_eq = [const]
    b_eq = [q0]

    # constraint 2: q(tw) = qw (waypoint)
    const = np.zeros(H + 2)
    const[H//2-1] = 1.0
    A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
    b_eq = np.insert(b_eq, len(b_eq), qw)

    # constraint 3: q(tf) = qf
    const = np.zeros(H + 2)
    const[H-1] = 1.0
    A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
    b_eq = np.insert(b_eq, len(b_eq), qf)

    # constraint 4: v(t0) = 0
    const = np.zeros(H + 2)
    const[0] = 1.0
    const[1] = -1.0
    A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
    b_eq = np.insert(b_eq, len(b_eq), 0.0)

    # constraint 5: v(tf) = 0
    const = np.zeros(H + 2)
    const[-4] = 1.0
    const[-3] = -1.0
    A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
    b_eq = np.insert(b_eq, len(b_eq), 0.0)
    return c, A_ub, b_ub, A_eq, b_eq


from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
dvrk_model = dvrkKinematics()

# pose information of pick & place
pos1 = np.array([0.14635528297124054, -0.02539699498070678, -0.15560356171404516])
rot1 = np.array([0.0, 0.0, 0.2985186426947612, 0.9544038034101067])
joint1 = dvrk_model.pose_to_joint(pos1, rot1)

pos2 = np.array([0.14635528297124054, -0.02539699498070678, -0.12])
rot2 = np.array([0.0, 0.0, 0.2985186426947612, 0.9544038034101067])
joint2 = dvrk_model.pose_to_joint(pos2, rot2)

pos3 = np.array([0.11511456574520817, -0.03390017669363312, -0.12])
rot3 = np.array([0.0, 0.0, 0.0, 1.0])
joint3 = dvrk_model.pose_to_joint(pos3, rot3)

print (joint1)
print (joint2)
print (joint3)

H = 300
a = 1
qs = []
for i in range(3):
    q0 = joint1[i]
    qw = joint2[i]
    qf = joint3[i]
    c, A_ub, b_ub, A_eq, b_eq = define_matrix(q0, qw, qf, horizon=H, abs_max_acc=a)
    result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    # print(result)

    dt1 = np.sqrt(result.x[-2])
    dt2 = np.sqrt(result.x[-1])
    print("dt1=", dt1)
    print("dt2=", dt2)
    t1 = np.arange(H//2) * dt1
    t2 = t1[-1] + (dt1 + dt2) / 2. + np.arange(H//2) * dt2
    t = np.concatenate((t1, t2), axis=0)
    # print(t1)
    # print(t2)
    plt.figure()
    plt.plot(t, result.x[:H], 'bo')

    # qs.append(result.x[:H])
    # v = sol[0][10:20]
    # v_pad = np.insert(v, 0, 0, axis=0)
    # v_pad = np.delete(v_pad, -1, axis=0)
    # a = v - v_pad


plt.show()

# plot
# t = np.arange(H)
# qs = np.array(qs)
# plot_joint(t, qs.T, qs.T)