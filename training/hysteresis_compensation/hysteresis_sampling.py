import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class HysteresisSimulation():
    def __init__(self, H_upper, H_lower):
        # load trajectory
        root = "/home/hwangmh/pycharmprojects/FLSpegtransfer/"
        dir = "experiment/0_trajectory_extraction/"
        self.q_des = np.load(root + dir + "training_traj_random.npy")[:500, 4]
        self.q_act = self.create_hysteresis(q_des=self.q_des, H_upper=H_upper, H_lower=H_lower)
        print ("data length: ", len(self.q_des))
        self.plot_hysteresis(self.q_des, self.q_act)

    def trajectory_sampling(self, num_sampling):
        q_des = np.random.uniform(-np.pi/2, np.pi/2, num_sampling)     # desired joint angle
        q_des_temp = np.insert(q_des, 0, 0)
        q_des_temp = np.delete(q_des_temp, -1)
        w_des = q_des - q_des_temp  # angular velocity
        return q_des, w_des

    def create_hysteresis(self, q_des, H_upper, H_lower):
        assert H_upper >= H_lower
        q_act = np.zeros_like(q_des)
        set_point = 0.0
        for i, qd in enumerate(q_des):
            if i == 0:
                q_act[i] = 0.0
                continue
            if set_point + H_lower < qd < set_point + H_upper:
                q_act[i] = q_act[i - 1]
            elif set_point + H_upper <= qd:
                q_act[i] = qd - H_upper
                set_point = qd - H_upper
            elif qd <= set_point + H_lower:
                q_act[i] = qd - H_lower
                set_point = qd - H_lower
        return q_act

    def plot_hysteresis(self, q_des, q_act):
        RMSE = []
        for i in range(6):
            RMSE = np.sqrt(np.sum((q_des - q_act) ** 2) / len(q_des))
        print("RMSE=", RMSE)

        # Create plot
        plt.style.use('seaborn-whitegrid')
        # plt.style.use('bmh')
        plt.rc('font', size=12)          # controls default text sizes
        plt.rc('axes', titlesize=20)     # fontsize of the axes title
        plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
        plt.rc('legend', fontsize=17)    # legend fontsize
        plt.rc('figure', titlesize=10)  # fontsize of the figure title

        t = range(len(q_des))
        ax = plt.subplot(111)
        plt.plot(t, np.rad2deg(q_des), 'b-')
        plt.plot(t, np.rad2deg(q_act), 'r-')
        plt.legend(loc='upper center', bbox_to_anchor=(0.9,2))
        ax.set_xticklabels([])
        plt.xlabel('samples')
        plt.ylabel('q ($^\circ$)')
        plt.show()


        # hysteresis plot
        plt.plot(np.rad2deg(q_des), np.rad2deg(q_act))
        plt.xlim([-80, 80])
        plt.ylim([-80, 80])
        plt.xlabel('q$_{des}(^\circ)$')
        plt.ylabel('q$_{act}(^\circ)$')
        plt.show()

if __name__ == "__main__":
    h = HysteresisSimulation(H_upper=np.deg2rad(10), H_lower=np.deg2rad(-10))
    np.save('q_des_hysteresis_sampled', h.q_des)
    np.save('q_act_hysteresis_sampled', h.q_act)