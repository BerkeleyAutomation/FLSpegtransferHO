import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BlockDetection2D import BlockDetection2D
from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
import numpy as np
from FLSpegtransfer.vision.PCLRegistration import PCLRegistration
import open3d as o3d

# load images
zivid = ZividCapture()
color = np.load('../../record/color_new.npy')
depth = np.load('../../record/depth_new.npy')
point = np.load('../../record/point_new.npy')
img_color, img_depth, img_point = zivid.img_crop(color, depth, point)
img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
pcl_board = BlockDetection3D.find_pegboard(img_color, img_point)

# load board model
pcl_model_board = o3d.io.read_point_cloud('peg_board_no_block.pcd')
o3d.visualization.draw_geometries([pcl_board, pcl_model_board])

# registration
reg = PCLRegistration()
Tcp = reg.registration(source=pcl_board, target=pcl_model_board, voxel_size=5, max_corr_dist=10)
Tpc = np.linalg.inv(Tcp)
np.save('Tpc.npy', Tpc)