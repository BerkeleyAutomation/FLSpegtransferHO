from FLSpegtransfer.traj_opt.QPOptimizer_2wp import PegMotionOptimizer_2wp
# from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.utils import CmnUtil as U
import numpy as np
import matplotlib.pyplot as plt

def cubic(q0, qf, v_max, a_max, t_step):
    num_axis = len(q0)
    q0 = np.array(q0)
    qf = np.array(qf)
    v_max = np.array(v_max)
    a_max = np.array(a_max)

    # v_max = 1.5*(qf-q0)/tf
    tf_vel = 1.5*(qf-q0) / v_max

    # a_max = 6*(qf-q0)/(tf**2)
    tf_acc = np.sqrt( abs(6*(qf-q0) / a_max))
    tf_Rn = np.maximum(tf_vel, tf_acc)  # tf for each axis (nx1 array)
    tf = max(tf_Rn)  # maximum scalar value among axes

    # Define coefficients
    a = -2 * (qf - q0) / (tf ** 3)
    b = 3 * (qf - q0) / (tf ** 2)
    c = np.zeros_like(a)
    d = q0

    # Calculate trajectories
    t = np.arange(start=0.0, stop=tf, step=t_step)
    joint = []
    for i in range(num_axis):
        # joint traj.
        q = a[i]*t**3 + b[i]*t**2 + c[i]*t + d[i]
        joint.append(q)
    joint = np.array(joint).T
    assert ~np.isnan(t).any()
    assert ~np.isnan(joint).any()
    return t, joint

# q0, qf could be cartesian coordinates or joint configurations
def LSPB(q0, qf, v_max, a_max, t_step):
    q0 = np.array(q0)
    qf = np.array(qf)
    v_max = np.array(v_max)
    a_max = np.array(a_max)
    num_axis = len(q0)

    # Design variables
    A = max(abs((qf-q0)/a_max))
    B = max(abs((qf-q0)/v_max))
    tb = A/B
    tf = B + tb
    if tf < 2*tb:
        tb = np.sqrt(A)
        tf = 2*tb

    # Define coefficients
    A = np.array([[tb**2, -tb, -1, 0.0, 0.0, 0.0],
                  [2*tb, -1, 0.0, 0.0, 0.0, 0.0],
                  [0.0, tf-tb, 1, -(tf-tb)**2, -(tf-tb), -1],
                  [0.0, 1.0, 0.0, -2*(tf-tb), -1, 0.0],
                  [0.0, 0.0, 0.0, 2*tf, 1.0, 0.0],
                  [0.0, 0.0, 0.0, tf**2, tf, 1.0]])
    b = np.block([[-q0], [np.zeros_like(q0)], [np.zeros_like(q0)], [np.zeros_like(q0)], [np.zeros_like(q0)], [qf]])
    coeff = np.linalg.inv(A).dot(b)
    a1 = coeff[0]
    a2 = coeff[1]
    b2 = coeff[2]
    a3 = coeff[3]
    b3 = coeff[4]
    c3 = coeff[5]

    # Calculate trajectories
    t = np.arange(start=0.0, stop=tf, step=t_step)
    t1 = t[t<tb]
    t2 = t[(tb<=t)&(t<tf-tb)]
    t3 = t[tf-tb<=t]
    joint = []
    for i in range(num_axis):
        # joint traj.
        traj1 = a1[i]*t1**2 + q0[i]
        traj2 = a2[i]*t2 + b2[i]
        traj3 = a3[i]*t3**2 + b3[i]*t3 + c3[i]
        q = np.concatenate((traj1, traj2, traj3))
        joint.append(q)
    joint = np.array(joint).T
    assert ~np.isnan(t).any()
    assert ~np.isnan(joint).any()
    return t, joint

def plot_trajectories(t, joint, t_step):
    vel = []
    acc = []
    joint = np.array(joint).T
    for q in joint:
        # velocity traj.
        q_prev = np.insert(q, 0, 0, axis=0)
        q_prev = np.delete(q_prev, -1, axis=0)
        qv = (q - q_prev)/t_step
        qv[0] = 0.0
        vel.append(qv)

        # acceleration traj.
        qv_prev = np.insert(qv, 0, 0, axis=0)
        qv_prev = np.delete(qv_prev, -1, axis=0)
        qa = (qv - qv_prev)/t_step
        qa[0] = 0.0
        acc.append(qa)

    joint = np.array(joint).T
    vel = np.array(vel).T
    acc = np.array(acc).T

    print ("maximum vel:", np.max(abs(vel), axis=0))
    print ("maximum acc:", np.max(abs(acc), axis=0))

    # Create plot
    # plt.title('joint angle')
    ax = plt.subplot(911)
    plt.plot(t, joint[:, 0], 'r.-')
    plt.ylabel('x (mm)')

    ax = plt.subplot(912)
    plt.plot(t, joint[:, 1], 'r.-')
    plt.ylabel('y (mm)')

    ax = plt.subplot(913)
    plt.plot(t, joint[:, 2], 'r.-')
    plt.ylabel('z (mm)')

    ax = plt.subplot(914)
    plt.plot(t, vel[:, 0], 'b.-')
    plt.ylabel('vx (mm)')

    ax = plt.subplot(915)
    plt.plot(t, vel[:, 1], 'b.-')
    plt.ylabel('vy (mm)')

    ax = plt.subplot(916)
    plt.plot(t, vel[:, 2], 'b.-')
    plt.ylabel('vz (mm)')

    ax = plt.subplot(917)
    plt.plot(t, acc[:, 0], 'g.-')
    plt.ylabel('ax (mm)')

    ax = plt.subplot(918)
    plt.plot(t, acc[:, 1], 'g.-')
    plt.ylabel('ay (mm)')

    ax = plt.subplot(919)
    plt.plot(t, acc[:, 2], 'g.-')
    plt.ylabel('az (mm)')
    plt.xlabel('t (sec)')
    plt.show()

from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
import time
dvrk_model = dvrkKinematics()

# pose information of pick & place
pose1 = np.array([0.14635528297124054, -0.02539699498070678, -0.15560356171404516, 0.6])
pose2 = np.array([0.14635528297124054, -0.02539699498070678, -0.12, 0.6])
pose3 = np.array([0.03511456574520817, -0.03390017669363312, -0.12, 0.0])
q0 = dvrk_model.pose_to_joint(pos=pose1[:3], rot=U.euler_to_quaternion([pose1[3], 0.0, 0.0]))
qw2 = dvrk_model.pose_to_joint(pos=pose2[:3], rot=U.euler_to_quaternion([pose2[3], 0.0, 0.0]))
qw1 = (1*np.array(q0) + 1*np.array(qw2))/2
qf = dvrk_model.pose_to_joint(pos=pose3[:3], rot=U.euler_to_quaternion([pose3[3], 0.0, 0.0]))
print (q0)
print (qw1)
print (qw2)
print (qf)

opt = PegMotionOptimizer_2wp()
pos, vel, acc, t = opt.optimize_motion(q0, qw1, qw2, qf, max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max, t_step=0.01, horizon=50, print_out=True, visualize=False)

dvrk = dvrkDualArm()
while True:
    dvrk.set_pose(pos1=pose1[:3], rot1=U.euler_to_quaternion([pose1[3], 0.0, 0.0]))
    dvrk.set_jaw(jaw1=[0.5])

    st = time.time()
    for joint in pos:
        # q = U.euler_to_quaternion([pose[3], 0.0, 0.0])
        # dvrk.set_pose(pos1=pose[:3], rot1=q, wait_callback=False)
        dvrk.set_joint(joint1=joint, wait_callback=False)
        time.sleep(0.01)
    print(time.time() - st)
    dvrk.set_pose(pos1=pose3[:3], rot1=U.euler_to_quaternion([pose3[3], 0.0, 0.0]))
    dvrk.set_jaw(jaw1=[0.5])
