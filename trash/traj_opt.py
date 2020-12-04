import numpy as np
import matplotlib.pyplot as plt

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

def define_objective_matrix(H):     # objective matrix to minimize squared sum of acceleration
    Z = np.zeros((H, H), dtype=np.float)
    temp = np.ones(H-2) * 2.0
    temp = np.pad(temp, (1, 1), 'constant', constant_values=(1, 1))
    D = np.diag(temp, k=0) + np.diag(-np.ones(H-1), k=1) + np.diag(-np.ones(H-1), k=-1)
    return np.block([[Z, Z], [Z, D]])

def define_constraint_matrix(q0, qf, dt, H):
    # constraint 1: q(n)-q(n-1) - v(n)*dt = 0
    A = (-np.diag(np.ones(H), k=0) + np.diag(np.ones(H-1), k=1))[:H-1]
    B = np.diag(np.ones(H-1)*(-dt), k=1)[:H-1]
    E = np.block([A,B])
    d = np.zeros(H-1).T

    # constraint 2: q(t0) = q0
    const = np.zeros(2*H)
    const[0] = 1.0
    E = np.insert(E, len(E), const, axis=0)
    d = np.insert(d, len(d), q0)

    # constraint 3: q(tf) = qf
    const = np.zeros(2*H)
    const[H-1] = 1.0
    E = np.insert(E, len(E), const, axis=0)
    d = np.insert(d, len(d), qf)

    # constraint 4: v(t0) = 0
    const = np.zeros(2 * H)
    const[H] = 1.0
    E = np.insert(E, len(E), const, axis=0)
    d = np.insert(d, len(d), 0.0)

    const = np.zeros(2 * H)
    const[H+1] = 1.0
    E = np.insert(E, len(E), const, axis=0)
    d = np.insert(d, len(d), 0.0)

    # constraint 5: v(tf) = 0
    const = np.zeros(2 * H)
    const[-1] = 1.0
    E = np.insert(E, len(E), const, axis=0)
    d = np.insert(d, len(d), 0.0)

    # constraint 6: waypoint
    const = np.zeros(2 * H)


    return E, d


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
dt = 0.1
qs = []
for i in range(6):
    q0 = joint1[i]
    qf = joint3[i]
    Q = define_objective_matrix(H)
    c = np.zeros(2*H)
    E, d = define_constraint_matrix(q0, qf, dt, H)

    A = np.block([[Q, E.T],[E, np.eye(np.shape(E)[0])]])
    b = np.concatenate((c,d), axis=0)
    sol = np.linalg.lstsq(A,b)
    qs.append(sol[0][:H])
    # v = sol[0][10:20]
    # v_pad = np.insert(v, 0, 0, axis=0)
    # v_pad = np.delete(v_pad, -1, axis=0)
    # a = v - v_pad

# plot
t = np.arange(H)
qs = np.array(qs)
plot_joint(t, qs.T, qs.T)