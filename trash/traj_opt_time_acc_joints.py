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


def define_matrix(q0, qw, qf, horizon):
    H = horizon

    # objective matrix to minimize smoothness
    v1 = np.zeros(6*H, dtype=np.float)
    v2 = np.ones(12, dtype=np.float)
    v = np.concatenate((v1,v2))
    Q = np.diag(v, k=0)
    c = np.zeros(6*H+12, dtype=np.float)

    # import pdb; pdb.set_trace()

    # Equality
    # constraint 1: q(n)-2*q(n-1)+q(n-2) - a*dt^2 = 0
    A1 = (np.diag(np.ones(H), k=0) + np.diag(-2*np.ones(H-1), k=1) + np.diag(np.ones(H-2), k=2))[:-2]
    Az = np.zeros_like(A1)
    As = np.zeros((6, H-2, 12))    # abs. max acceleration for each joint
    for i in range(6):
        As[i, :H//2-1, i] = -1
        As[i, H//2-1:, i+6] = -1
    A_eq = np.block([[A1, Az, Az, Az, Az, Az, As[0]],
                     [Az, A1, Az, Az, Az, Az, As[1]],
                     [Az, Az, A1, Az, Az, Az, As[2]],
                     [Az, Az, Az, A1, Az, Az, As[3]],
                     [Az, Az, Az, Az, A1, Az, As[4]],
                     [Az, Az, Az, Az, Az, A1, As[5]]])
    b_eq = np.zeros(np.shape(A_eq)[0])

    # constraint 1: q(t0) = q0
    for i in range(6):
        const = np.zeros(6*H + 12)
        const[i*H + 0] = 1.0
        A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
        b_eq = np.insert(b_eq, len(b_eq), q0[i])

    # constraint 2: q(tw) = qw (waypoint)
    for i in range(6):
        const = np.zeros(6*H + 12)
        const[i*H + H//2 - 1] = 1.0
        # const[i*H + 10 - 1] = 1.0
        A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
        b_eq = np.insert(b_eq, len(b_eq), qw[i])

    # constraint 3: q(tf) = qf
    for i in range(6):
        const = np.zeros(6*H + 12)
        const[i*H + H-1] = 1.0
        A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
        b_eq = np.insert(b_eq, len(b_eq), qf[i])

    # constraint 4: v(t0) = 0
    for i in range(6):
        const = np.zeros(6*H + 12)
        const[i*H + 0] = 1.0
        const[i*H + 1] = -1.0
        A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
        b_eq = np.insert(b_eq, len(b_eq), 0.0)

    # constraint 5: v(tf) = 0
    for i in range(6):
        const = np.zeros(6*H + 12)
        const[i*H + H-2] = 1.0
        const[i*H + H-1] = -1.0
        A_eq = np.insert(A_eq, len(A_eq), const, axis=0)
        b_eq = np.insert(b_eq, len(b_eq), 0.0)
    return Q, c, A_eq, b_eq


def solve_QP(Q, c, A_eq, b_eq):
    A = np.block([[Q, A_eq.T], [A_eq, np.eye(np.shape(A_eq)[0])]])
    b = np.concatenate((-c, b_eq))
    result = np.linalg.lstsq(A, b)
    return result[0][:np.shape(Q)[0]]


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
# joint1[3:] = -joint1[3:]
# joint2[3:] = -joint2[3:]
# joint3[3:] = -joint3[3:]

dt1= 0.008091745397496325
dt2= 0.016512854654590493


H = 50
print ("dt1:", H*dt1/(dt1+dt2))
print ("dt2:", H*dt2/(dt1+dt2))
a = [30, 10, 10, 30, 10, 100]
qs = []
q0 = joint1
qw = joint2
qf = joint3

print ("q0:", q0)
print ("qw:", qw)
print ("qf:", qf)

import time
st = time.time()
Q, c, A_eq, b_eq = define_matrix(q0, qw, qf, horizon=H)
result = solve_QP(Q, c, A_eq, b_eq)
x = result[:6*H]
adt1 = result[-12:-6]
adt2 = result[-6:]
print (adt1)
print (adt2)
# print(x)
# print(adt)
print ("solving time:", time.time() - st)


# dt1 = np.sqrt(result.x[-2])
# dt2 = np.sqrt(result.x[-1])
# print("dt1=", dt1)
# print("dt2=", dt2)
#
# t1 = np.arange(H//2) * dt1
# t2 = t1[-1] + (dt1 + dt2) / 2. + np.arange(H//2) * dt2
# t = np.concatenate((t1, t2), axis=0)
# print(t1)
# print(t2)

a = np.array(a)
dt1 = np.sqrt(max(abs(adt1/a)))
dt2 = np.sqrt(max(abs(adt2/a)))
print ("time: ", dt1, dt2)

# import pdb; pdb.set_trace()

qs = x.reshape(-1,H).T
# print (qs)
# t =
t = range(len(qs))
plot_joint(t, qs, qs)

# qs.append(result.x[:H])
# v = sol[0][10:20]
# v_pad = np.insert(v, 0, 0, axis=0)
# v_pad = np.delete(v_pad, -1, axis=0)
# a = v - v_pad
# plot
