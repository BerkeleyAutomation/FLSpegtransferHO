import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def plot_joint(t, q_des, q_act):
    # Create plot
    # plt.title('joint angle')
    ax = plt.subplot(611)
    plt.plot(t, q_des[:,0]*180./np.pi, 'bo-')
    plt.plot(t, q_act[:, 0] * 180. / np.pi, 'ro-')
    plt.legend(loc='upper center', bbox_to_anchor=(0.9,2))
    # plt.legend(['q_des', 'q_grey', 'q_red'], ncol=3, bbox_to_anchor=(0.5, 1), loc="lower center")
    # plt.ylim([35, 62])
    ax.set_xticklabels([])
    plt.ylabel('q0 ($^\circ$)')

    ax = plt.subplot(612)
    plt.plot(t, q_des[:,1]*180./np.pi, 'bo-', t, q_act[:, 1] * 180. / np.pi, 'ro-')
    # plt.ylim([-10, 12])
    ax.set_xticklabels([])
    plt.ylabel('q2 ($^\circ$)')

    ax = plt.subplot(613)
    plt.plot(t, q_des[:, 2], 'bo-', t, q_act[:, 2], 'ro-')
    # plt.ylim([0.14, 0.23])
    ax.set_xticklabels([])
    plt.ylabel('q3 (m)')

    ax = plt.subplot(614)
    plt.plot(t, q_des[:, 3]*180./np.pi, 'bo-', t, q_act[:, 3]*180./np.pi, 'ro-')
    # plt.ylim([-90, 70])
    ax.set_xticklabels([])
    plt.ylabel('q4 ($^\circ$)')

    ax = plt.subplot(615)
    plt.plot(t, q_des[:, 4]*180./np.pi, 'bo-', t, q_act[:, 4]*180./np.pi, 'ro-')
    # plt.ylim([-60, 60])
    ax.set_xticklabels([])
    plt.ylabel('q5 ($^\circ$)')

    plt.subplot(616)
    plt.plot(t, q_des[:, 5]*180./np.pi, 'bo-', t, q_act[:, 5]*180./np.pi, 'ro-')
    # plt.ylim([-60, 60])
    # plt.legend(['desired', 'actual'])
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('sample number')
    plt.show()


def define_matrix(q0, qw, qf, abs_max_acc, horizon, n):
    H = horizon
    a = abs_max_acc

    # objective matrix to minimize time
    c = np.zeros(6*H+2, dtype=np.float)
    c[-1] = H-n
    c[-2] = n

    # Inequality
    # constraint 1: -a_max <= (q(n)-2*q(n-1)+q(n-2))/dt^2 <= a_max
    A1 = (np.diag(np.ones(H), k=0) + np.diag(-2*np.ones(H-1), k=1) + np.diag(np.ones(H-2), k=2))[:-2]
    A2 = (np.diag(-np.ones(H), k=0) + np.diag(2*np.ones(H-1), k=1) + np.diag(-np.ones(H-2), k=2))[:-2]
    Az = np.zeros_like(A1)
    As = np.zeros((6, H-2, 2))    # abs. max acceleration for each joint
    for i in range(6):
        As[i, :n-1, 0] = -a[i]
        As[i, n-1:, 1] = -a[i]

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
        const = np.zeros(6*H + 2)
        const[i*H + 0] = 1.0
        A_eq.append(const)
        b_eq.append(q0[i])

    # constraint 2: q(tw) = qw (waypoint)
    for i in range(6):
        const = np.zeros(6*H+2)
        # const[i*H + H//2 - 1] = 1.0
        const[i*H + n - 1] = 1.0
        # import pdb; pdb.set_trace()
        A_eq.append(const)
        b_eq.append(qw[i])

    # constraint 3: q(tf) = qf
    for i in range(6):
        const = np.zeros(6*H+2)
        const[i*H + H-1] = 1.0
        A_eq.append(const)
        b_eq.append(qf[i])

    # constraint 4: v(t0) = 0
    for i in range(6):
        const = np.zeros(6*H+2)
        const[i*H + 0] = 1.0
        const[i*H + 1] = -1.0
        A_eq.append(const)
        b_eq.append(0.0)

    # constraint 5: v(tf) = 0
    for i in range(6):
        const = np.zeros(6*H+2)
        const[i*H + H-2] = 1.0
        const[i*H + H-1] = -1.0
        A_eq.append(const)
        b_eq.append(0.0)
    A_eq = np.array(A_eq)
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

joint1 = np.array(joint1)
joint2 = np.array(joint2)
joint3 = np.array(joint3)
joint1[3:] = -joint1[3:]
joint2[3:] = -joint2[3:]
joint3[3:] = -joint3[3:]

# print (joint1)
# print (joint2)
# print (joint3)

H = 100
a = [3, 1, 1, 3, 1, 3]
qs = []
q0 = joint1
qw = joint2
qf = joint3

number = np.arange(10,90)
ttt = []
# n = 50
for n in number:
    # import time
    c, A_ub, b_ub, A_eq, b_eq = define_matrix(q0, qw, qf, horizon=H, abs_max_acc=a, n=n)
    # st = time.time()
    result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    print(result)
    # print ("solving time:", time.time() - st)

    print ("obj: ", result.x[-2] + result.x[-1])
    dt1 = np.sqrt(result.x[-2])
    dt2 = np.sqrt(result.x[-1])
    print("dt1=", dt1)
    print("dt2=", dt2)
    total = n*dt1+(H-n)*dt2
    print ("total time=", total)
    print (H*dt1*n/total)
    print (H*dt2*(H-n)/total)


    t1 = np.arange(n) * dt1
    t2 = t1[-1] + (dt1+dt2)/2. + np.arange(H-n) * dt2
    t = np.concatenate((t1, t2), axis=0)
    # print(t1)
    # print(t2)
    print ()
    print ("total time: ", t[-1])
    print ("ratio1: ", n*dt1/t[-1])
    print ("ratio2: ", (H-n)*dt2/t[-1])
    print ("border: ", n*dt1)
    qs = result.x[:6*H].reshape(-1,H).T
    ttt.append(t[-1])


# print (qs)
# plot_joint(t, qs, qs)
print (ttt)
plt.figure()
plt.plot(number, ttt)
plt.show()