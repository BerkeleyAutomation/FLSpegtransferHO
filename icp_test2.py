import pygicp
import numpy as np


# target = [[0.0, 1.0, 0.0], [20, 2, 3.0], [40, 5, 6], [10, 2, 3]] # Nx3 numpy array
# source = [[0.0, 1.0, 0.0], [5, 2, 3.0], [40, 5, 10], [2, 2, 3]] # Mx3 numpy array
target = np.load("pnt_blk.npy")
source = np.load("pnt_model.npy")

# 1. function interface
import time

st = time.time()
matrix = pygicp.align_points(target, source)
print (time.time() - st)
print(matrix)

# optional arguments
# initial_guess               : Initial guess of the relative pose (4x4 matrix)
# method                      : GICP, VGICP, VGICP_CUDA, or NDT_CUDA
# downsample_resolution       : Downsampling resolution (used only if positive)
# k_correspondences           : Number of points used for covariance estimation
# max_correspondence_distance : Maximum distance for corresponding point search
# voxel_resolution            : Resolution of voxel-based algorithms
# neighbor_search_method      : DIRECT1, DIRECT7, DIRECT27, or DIRECT_RADIUS
# neighbor_search_radius      : Neighbor voxel search radius (for GPU-based methods)
# num_threads                 : Number of threads


# 2. class interface
# you may want to downsample the input clouds before registration
# target = pygicp.downsample(target, 0.25)
# source = pygicp.downsample(source, 0.25)

# pygicp.FastGICP has more or less the same interfaces as the C++ version
gicp = pygicp.FastGICP()
gicp.set_input_target(target)
gicp.set_input_source(source)
matrix = gicp.align()
print (matrix)

# optional
gicp.set_num_threads(4)
gicp.set_max_correspondence_distance(1.0)
gicp.get_final_transformation()
gicp.get_final_hessian()