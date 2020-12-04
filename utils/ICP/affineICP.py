import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class affineICP:
    @classmethod
    def nearest_neighbor(cls, src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()


    @classmethod
    def best_fit_transform(cls, source, target):
        # assert source.shape == target.shape

        # get number of dimensions
        m = source.shape[1]

        # translate points to their centroids
        centroid_S = np.mean(source, axis=0)
        centroid_T = np.mean(target, axis=0)
        SS = source - centroid_S
        TT = target - centroid_T

        # affine transform
        temp1 = TT.T.dot(SS)
        temp2 = np.linalg.inv(SS.T.dot(SS))
        A = temp1.dot(temp2)

        # translation
        t = centroid_T - A.dot(centroid_S)

        # homogeneous transformation
        T = np.identity(m + 1)
        T[:m, :m] = A
        T[:m, m] = t
        return T, A, t

    @classmethod
    def icp(cls, A, B, init_pose=None, max_iterations=50, tolerance=0.001):
        # assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1, A.shape[0]))
        dst = np.ones((m + 1, B.shape[0]))
        src[:m, :] = np.copy(A.T)
        dst[:m, :] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0
        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = affineICP.nearest_neighbor(src[:m, :].T, dst[:m, :].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = affineICP.best_fit_transform(src[:m, :].T, dst[:m, indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T, _, _ = affineICP.best_fit_transform(A, src[:m, :].T)

        return T, distances, i