from motion.dvrkDualArm import dvrkDualArm
from vision.ZividCapture import ZividCapture
import utils.CmnUtil as U
import cv2
import numpy as np

class FabricFoldingMotion:
	def __init__(self):
		# instances
		self.dvrk = dvrkDualArm()

		# load transform
		self.Trc1 = np.load('calibration_files/Trc_overhead_PSM1.npy')
		self.Trc2 = np.load('calibration_files/Trc_overhead_PSM2.npy')

		# Motion variables
		self.pos_org1 = [0.020, 0.0, -0.095]
		self.rot_org1 = [0.0, 0.0, 0.0, 1.0]
		self.pos_neutral1 = [0.1, 0.0, -0.1]
		self.rot_neutral1 = [0.0, 0.0, 0.0, 1.0]

		self.height_grasp_offset = -0.005
		self.height_ready = -0.120
		self.height_drop = -0.137

		self.jaw_opening = [np.deg2rad(40)]
		self.jaw_closing = [np.deg2rad(-10)]

	# self.dvrk.set_pose(pos1=[x,y,z], rot1=[qx,qy,qz,qw], pos2=[x,y,z], rot2=[qx,qy,qz,qw])
	# self.dvrk.set_jaw(jaw1=[jaw_] ,jaw2=[])

	def transform_cam2robot(self, point_cam, which_arm='PSM1'):  # input: 3d point w.r.t camera coordinate (m)
		point_cam = np.array(point_cam)*0.001
		if which_arm == 'PSM1':
			Trc = self.Trc1
		else:
			Trc = self.Trc2
		R = Trc[:3, :3]
		t = Trc[:3, -1]
		return R.dot(point_cam.T).T + t.T

	def mask_image(self, img_color, img_point):
		lower_blue = np.array([120 - 50, 50, 40])
		upper_blue = np.array([120 + 50, 255, 255])
		hsv_range = [lower_blue, upper_blue]

		# 2D color masking
		img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
		img_masked = cv2.inRange(img_hsv, hsv_range[0], hsv_range[1])

		# color masking
		con1 = (img_masked == 255)
		arg1 = np.argwhere(con1)
		pnt1 = img_point[con1]

		# remove nan
		con2 = (~np.isnan(pnt1).any(axis=1))
		arg2 = np.argwhere(con2)
		pnt2 = pnt1[con2]

		# depth masking
		depth_range = [800, 900]
		con3 = (pnt2[:, 2] > depth_range[0]) & (pnt2[:, 2] < depth_range[1])
		arg3 = np.argwhere(con3)

		# creat mask where the above conditions hold
		arg_mask = np.squeeze(arg1[arg2[arg3]])
		mask = np.zeros_like(img_masked)
		mask[arg_mask[:, 0], arg_mask[:, 1]] = 255
		return mask

	def find_3d_point(self, pick_point, img_color, img_point, visualization=False):
		masked = self.mask_image(img_color, img_point)
		if visualization:
			cv2.imshow("", masked)
			cv2.waitKey(0)
		dx = 5
		dy = 5
		arg = np.argwhere(masked==255)
		con1 = (pick_point[0]-dx < arg[:,0]) & (arg[:,0] < pick_point[0]+dx)
		con2 = (pick_point[1]-dy < arg[:,1]) & (arg[:,1] < pick_point[1]+dy)
		points = arg[con1 & con2]	# pixels that is not Nan
		point_selected = points[0]
		if visualization:
			cv2.circle(img_color, (point_selected[1], point_selected[0]), 5, (0, 255, 0), 2)
			cv2.imshow("", img_color)
			cv2.waitKey(0)
		return img_point[point_selected[0], point_selected[1]]

	def move_origin(self):
		self.dvrk.set_pose(pos1=self.pos_org1, rot1=self.rot_org1)
		self.dvrk.set_jaw(jaw1=self.jaw_closing)

	def pick_and_pull(self, pos_pock, pos_pull, rot):  # using PSM1
		# go neutral configuratoin
		self.dvrk.set_pose(pos1=self.pos_neutral1, rot1=self.rot_neutral1)

		# go above pick-spot
		rot1 = np.deg2rad([rot, 0.0, 0.0])
		rot1 = U.euler_to_quaternion(rot1)
		self.dvrk.set_pose(pos1=[pos_pick[0], pos_pick[1], self.height_ready], rot1=rot1)
		self.dvrk.set_jaw(jaw1=self.jaw_opening)

		# go down toward the knot & close jaw
		rot1 = np.deg2rad([rot, 0.0, 0.0])
		rot1 = U.euler_to_quaternion(rot1)
		self.dvrk.set_pose(pos1=[pos_pick[0], pos_pick[1], pos_pick[2]+self.height_grasp_offset], rot1=rot1)
		self.dvrk.set_jaw(jaw1=self.jaw_closing)

		# go up
		self.dvrk.set_pose(pos1=[pos_pick[0], pos_pick[1], pos_pull[2]], rot1=rot1)

		# pull
		rot1 = np.deg2rad([rot, 0.0, 0.0])
		rot1 = U.euler_to_quaternion(rot1)
		self.dvrk.set_pose(pos1=[pos_pull[0], pos_pull[1], pos_pull[2]], rot1=rot1)
		self.dvrk.set_jaw(jaw1=self.jaw_opening)

		# go neutral configuratoin
		self.dvrk.set_pose(pos1=self.pos_neutral1, rot1=self.rot_neutral1)


if __name__ == "__main__":
	motion = FabricFoldingMotion()
	zivid = ZividCapture(which_camera="overhead")
	zivid.start()
	while True:
		motion.move_origin()

		img_color, img_depth, img_point = zivid.capture_3Dimage(color='BGR')
		cv2.imshow("", img_color)
		cv2.waitKey(0)

		# pick pixel
		pick_pix = [500, 1250]	# pick_point in pixel
		pick_3d_cam = motion.find_3d_point(pick_pix, img_color, img_point, visualization=True)	# pick point in 3d (x,y,z)
		pick_3d_rob = motion.transform_cam2robot(pick_3d_cam, which_arm='PSM1')
		pos_pick = pick_3d_rob
		pos_pull = pick_3d_rob + np.array([0.01, 0.01, 0.03])
		motion.pick_and_pull(pos_pick, pos_pull, 0)
