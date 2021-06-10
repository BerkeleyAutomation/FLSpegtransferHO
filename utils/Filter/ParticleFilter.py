from FLSpegtransfer.vision.ToolPoseEstimation import ToolPoseEstimation
import numpy as np
from numpy.random import uniform, randn
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
import time
from FLSpegtransfer.vision.AlliedVisionUtils import AlliedVisionUtils


class ParticleFilter:    # Particle filter for rob2cam calibration
    def __init__(self):
        self.initialized = False
        self.particles = np.array([])
        self.weights = np.array([])
        self.N = 0
        self.av_utils = AlliedVisionUtils()

    @classmethod
    def estimate(cls, particles, weights):
        """returns mean and variance of the weighted particles"""
        mean = np.average(particles, weights=weights, axis=0)
        var = np.average((particles - mean) ** 2, weights=weights, axis=0)
        return mean, var

    @classmethod
    def simple_resample(cls, particles, weights):
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.  # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.random(N))

        # resample according to indexes
        particles[:] = particles[indexes]
        weights.fill(1.0 / N)

    @classmethod
    def neff(cls, weights):
        return 1. / np.sum(np.square(weights))

    @classmethod
    def resample_from_index(cls, particles, weights, indexes):
        particles[:] = particles[indexes]
        weights.resize(len(particles))
        weights.fill(1.0 / len(weights))

    def init(self, x_init, lower_limit, upper_limit, N=5000):    # this function is called only once at first
        """
        :param x_init: initial guess of state
        :param N: number of particles
        :return:
        """
        # self.std = std
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        if not self.initialized:
            # create particles and weights
            # self.particles = x_init + (randn(N, len(x_init)) * std)   # create Gaussian particles
            self.particles = x_init + uniform(lower_limit, upper_limit, size=(N, len(x_init)))  # create uniform particles
            self.weights = np.ones(N) / N
            self.N = N
            self.initialized = True

    def update(self, z, arg, u, visualize=False):
        """
        :param z: state measured by sensor
        :param arg: which keypoints to select
        :param u: control input
        :param landmarks: criteria for updating weight
        :param sensor_std_err: std error of sensor
        :param visualize: True or False
        :return:
        """

        """
        predict particles: move according to control input u with noise Q
        """
        N = len(self.particles)     # number of particles
        n = np.shape(self.particles)[1]     # number of variables to estimate
        # self.particles += np.array(u) + (randn(N, n) * self.std)
        self.particles += np.array(u) + uniform(self.lower_limit, self.upper_limit, size=(N, n))

        """
        update weights
        """
        kps_model = ToolPoseEstimation.get_keypoints_model_q567(self.particles)[:, arg, :]  # w.r.t cam frame
        kps_model_tr = ToolPoseEstimation.get_rigid_transform(kps_model, [z[arg]])
        distance = np.linalg.norm(kps_model_tr - z[arg], axis=2)
        dist_sum = np.sum(distance, axis=1)  # various metrics can be used here (ex) RMS, chamfer dist, etc
        tau = 0.001     # sensitivity factor
        self.weights *= np.exp(-dist_sum/tau)
        # weights *= scipy.stats.norm(distance, R).pdf(z[i])
        # self.weights *= scipy.stats.norm(dist_sum, 0.01).pdf(0.0)
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize


        """
        resample if too few effective particles
        """
        if self.neff(self.weights) < self.N/2:
            indexes = systematic_resample(self.weights)
            self.resample_from_index(self.particles, self.weights, indexes)
            assert np.allclose(self.weights, 1 / self.N)
        mu, var = self.estimate(self.particles, self.weights)

        if visualize:
            plt.scatter(self.particles[:, 0], self.particles[:, 1], color='k', marker=',', s=1)
            p1 = plt.scatter(z[0], z[1], marker='+', color='r', s=180, lw=3)
            p2 = plt.scatter(mu[0], mu[1], marker='s', color='b')
            plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
            xlim = (-0.1, 0.1)
            ylim = (-0.1, 0.1)
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.show()
        return mu, var


if __name__ == "__main__":
    from numpy.random import seed
    pf = ParticleFilter()
    seed(2)
    robot_pos = np.array([0.0, 0.0])
    pf.init(x_init=None, N=5000)
    while True:
        robot_pos += [1.0, 1.0]
        pf.update(z=robot_pos, visualize=True)

    # seed(2); pf.run_pf(N=5000, iters=8, plot_particles=True, xlim=(0, 8), ylim=(0, 8))
    # seed(2); pf.run_pf(N=100000, iters=8, plot_particles=True, xlim=(0, 8), ylim=(0, 8))
    # seed(6); pf.run_pf(N=5000, plot_particles=True, ylim=(-20, 20))
    # seed(6); pf.run_pf(N=5000, plot_particles=True, initial_x=(1, 1, np.pi / 4))