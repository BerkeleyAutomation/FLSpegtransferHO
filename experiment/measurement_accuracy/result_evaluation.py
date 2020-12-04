import numpy as np
from sklearn.metrics import mean_squared_error

# Load experimental data
# total_modeled = np.load("total_modeled_new.npy")
# total_measured = np.load("total_measured_new.npy")
total_modeled = np.load("total_modeled_old.npy")
total_measured = np.load("total_measured_old.npy")
total_modeled = np.array(total_modeled).ravel()
total_measured = np.array(total_measured).ravel()

# Evaluate data
RMSE = np.sqrt(mean_squared_error(total_modeled, total_measured))
error = total_modeled - total_measured
avr = np.average(error, axis=0)
error_abs = abs(total_modeled - total_measured)
SD = np.std(error_abs)
print ("Sphere Detection Error Evaluation")
print ("Number of samples: ", len(total_measured))
print("RMSE_total=", RMSE, '(mm)')
print("Average_total=", avr, '(mm)')
print("SD_total=", SD)
print("Median_pos_error=", np.median(error), '(mm)')
print("Max_pos_error=", np.max(error), '(mm)')
print("Min_pos_error=", np.min(error), '(mm)')


# Plot error graph
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.style.use('bmh')
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=17)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title

t = range(len(total_modeled))
plt.figure()
plt.plot(t, error, 'o-', markersize=3)
plt.xlabel("Samples")
plt.ylabel("Error (mm)")
plt.xticks(np.arange(0, 122, step=10))
bins = np.arange(-0.8, 0.8, 0.08)
plt.figure()
plt.xlabel("Error (mm)")
plt.ylabel("Frequency")
plt.hist(error, bins=bins)
plt.show()


# Evaluate the error effects on joint angle estimation
from FLSpegtransfer.vision.BallDetectionRGBD import BallDetectionRGBD
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
BD = BallDetectionRGBD()

def plot_joint(t, q_des, q_act):
    t = np.array(t)
    q_des = np.array(q_des)
    q_act = np.array(q_act)

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


def estimate_joint_angles(pbr, pbg, pbb, pby):
    # Find tool(wrist) position
    pt = BD.find_tool_position(pbr[0], pbr[1])  # tool position of pitch axis
    pt = np.array(pt) * 0.001  # (m)
    pt = BD.Rrc.dot(pt) + BD.trc
    q1, q2, q3 = dvrkKinematics.ik_position_straight(pt)

    # Find tool orientation
    Rm = BD.find_tool_orientation(pbr[2], pbg, pbb, pby)  # orientation of the marker
    q4, q5, q6 = dvrkKinematics.ik_orientation(q1, q2, Rm)
    return [q1,q2,q3,q4,q5,q6]


def add_error(pos, max, nb_sample, visualize=False):
    N = nb_sample

    phi = np.random.uniform(low=0, high=2*np.pi, size=N)
    costheta = np.random.uniform(low=-1., high=1., size=N)
    u = np.random.uniform(low=0., high=1., size=N)
    theta = np.arccos(costheta)
    r = max * u**(1/3)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    p = np.concatenate((x,y,z)).reshape(3,-1).T

    if visualize:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(nb_sample):
            ax.scatter(p[i,0], p[i,1], p[i,2])
        plt.show()
    return p + pos


# Joint angle estimation
D = 1700
pbr = np.array([[0., 0., D], [0., 0., D+50.], [-35., 0., D+50+39.52]])           # define ground-truth value
pbg = np.array([0., 35., D+50+39.52])
pbb = np.array([35., 0., D+50+39.52])
pby = np.array([0., -35., D+50+39.52])
q_act = estimate_joint_angles(pbr, pbg, pbb, pby)
print ("")
print ("ground-truth joint angle=", q_act)
print ("")


# Random sampling
mu = np.mean(error)
sigma = np.std(error)
max = np.max(error_abs)
min = np.min(error)
N = 1000
pbr1 = add_error(pos=pbr[0], max=max, nb_sample=N)
pbr2 = add_error(pos=pbr[1], max=max, nb_sample=N)
pbr3 = add_error(pos=pbr[2], max=max, nb_sample=N)
pbg = add_error(pos=pbg, max=max, nb_sample=N)
pbb = add_error(pos=pbb, max=max, nb_sample=N)
pby = add_error(pos=pby, max=max, nb_sample=N)

q_err = []
for r1,r2,r3,g,b,y in zip(pbr1, pbr2, pbr3, pbg, pbb, pby):
    q_noise = estimate_joint_angles([r1,r2,r3],g,b,y)
    q_err.append(np.array(q_act) - np.array(q_noise))

q_err = np.array(q_err)
t = range(len(q_err))
plot_joint(t, q_err, q_err)

# plt.hist(q_err[:,0], bins='auto')
# plt.show()

q_err[:,0] *= 180/np.pi
q_err[:,1] *= 180/np.pi
q_err[:,2] *= 1000
q_err[:,3] *= 180/np.pi
q_err[:,4] *= 180/np.pi
q_err[:,5] *= 180/np.pi

RMSE = np.sqrt(np.sum((q_err) ** 2, axis=0) / len(q_err))
SD = np.std(q_err, axis=0)
avr = np.average(q_err, axis=0)
max = np.max(q_err, axis=0)
min = np.min(q_err, axis=0)
median = np.median(q_err, axis=0)

print("")
print("Joint Error Evaluation")
print("RMSE=", RMSE)
print("Average=", avr)
print("SD=", SD)
print("Median=", median)
print("Max=", max)
print("Min=", min)
