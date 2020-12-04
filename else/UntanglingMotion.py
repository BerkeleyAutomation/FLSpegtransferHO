from motion.dvrkDualArm import dvrkDualArm
#from vision.ZividCapture import ZividCapture
import utils.CmnUtil as U
import cv2
import numpy as np


class UntanglingMotion:
	def __init__(self):
		# instances
		self.dvrk = dvrkDualArm()
		#self.zivid = ZividCapture(which_camera="overhead")
		#self.zivid.start()		
		
		# load transform
		self.Trc1 = np.load('calibration_files/Trc1.npy')
		self.Trc2 = np.load('calibration_files/Trc2.npy')
		
		# Motion variables
		self.pos_org1 = [0.080, 0.0, -0.095]
		self.rot_org1 = [0.0, 0.0, 0.0, 1.0]
		self.pos_org2 = [-0.080, 0.0, -0.095]
		self.rot_org2 = [0.0, 0.0, 0.0, 1.0]

		self.height_grasp_offset_above = +0.003
		self.height_grasp_offset_below = -0.008
		self.height_ready = -0.120
		self.height_drop = -0.137

		self.jaw_opening = [np.deg2rad(60)]
		self.jaw_closing = [np.deg2rad(10)]
		
		#self.dvrk.set_pose(pos1=[x,y,z], rot1=[qx,qy,qz,qw], pos2=[x,y,z], rot2=[qx,qy,qz,qw])
#		self.dvrk.set_jaw(jaw1=[jaw_] ,jaw2=[])

	
	def transform_cam2robot(self, point_cam, which_arm='PSM1'):	# input: 3d point w.r.t camera coordinate (m)
		point_cam = np.array(point_cam)
		if which_arm == 'PSM1':
			Trc = self.Trc1
		else:
			Trc = self.Trc2		
		R = Trc[:3,:3]
		t = Trc[:3,-1]
		return R.dot(point_cam.T).T + t.T
	        
	        
	def move_origin(self):
		self.dvrk.set_pose(pos1=self.pos_org1, rot1=self.rot_org1, pos2=self.pos_org2, rot2=self.rot_org2)
		self.dvrk.set_jaw(jaw1=self.jaw_closing, jaw2=self.jaw_closing)
		
		
	def hold_knot(self, pos1, rot1):	# using PSM1
		# go above the knot to hold
		pos = [pos1[0], pos1[1], self.height_ready]
		rot = np.deg2rad([rot1, 0.0, 0.0])
		rot = U.euler_to_quaternion(rot)
		self.dvrk.set_pose(pos1=pos, rot1=rot)	
		self.dvrk.set_jaw(jaw1=self.jaw_opening)
		
		# go down toward the knot & close jaw
		pos = [pos1[0], pos1[1], pos1[2]]
		rot = np.deg2rad([rot1, 0.0, 0.0])
		rot = U.euler_to_quaternion(rot)
		self.dvrk.set_pose(pos1=pos, rot1=rot)
		self.dvrk.set_jaw(jaw1=self.jaw_closing)

	        	        
	def grasp_knot(self, pos2, rot2):	# using PSM2
		# go above the knot to pick up
		pos = [pos2[0], pos2[1], self.height_ready]
		rot = np.deg2rad([rot2, 0.0, 0.0])
		rot = U.euler_to_quaternion(rot)
		self.dvrk.set_pose(pos2=pos, rot2=rot)
		self.dvrk.set_jaw(jaw2=self.jaw_opening)
		
		# go down toward the knot & close jaw
		pos = [pos2[0], pos2[1], pos2[2]]
		rot = np.deg2rad([rot2, 0.0, 0.0])
		rot = U.euler_to_quaternion(rot)
		self.dvrk.set_pose(pos2=pos, rot2=rot)
		self.dvrk.set_jaw(jaw2=self.jaw_closing)
		

	def pull_knot(self, pos2, rot2):	# using PSM2
		# pull knot
		pos = [pos2[0], pos2[1], pos2[2]]
		rot = np.deg2rad([rot2, 0.0, 0.0])
		rot = U.euler_to_quaternion(rot)
		self.dvrk.set_pose(pos2=pos, rot2=rot)
		
		
	def release(self):
		self.dvrk.set_jaw(jaw1=self.jaw_opening, jaw2=self.jaw_opening)
		pose, _ = self.dvrk.get_pose()
		pos, rot, jaw = pose
		pos[2] = self.height_ready		
		self.dvrk.set_pose(pos1=pos, rot1=rot)


if __name__ == "__main__":
	motion = UntanglingMotion()
	while True:
		point_cam = [0.0, 0.0, 0.9]	# output 3d points w.r.t camera coordinate in (m)
		point_rob = motion.transform_cam2robot(point_cam, which_arm='PSM1')
		motion.move_origin()
		motion.hold_knot(point_rob, 30)
		motion.grasp_knot([-0.15, -0.0157, -0.16], -40)
		motion.pull_knot([-0.13, -0.03, -0.14], -40)
		motion.release()
