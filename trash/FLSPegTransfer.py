from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.BallDetectionRGBD import BallDetectionRGBD
from FLSpegtransfer.vision.GraspingPose3D import GraspingPose3D
from FLSpegtransfer.vision.VisualizeDetection import VisualizeDetection
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration
from FLSpegtransfer.motion.dvrkPegTransferMotionSingleArm import dvrkPegTransferMotion
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.path import *
import numpy as np

class FLSPegTransfer():
    def __init__(self, which_camera):
        # load transform
        self.Trc = np.load(root + 'calibration_files/Trc_' + which_camera + '_PSM1.npy')  # robot to camera
        self.Tpc = np.load(root + 'calibration_files/Tpc_' + which_camera + '.npy')  # pegboard to camera

        # import modules
        self.zivid = ZividCapture(which_camera=which_camera)
        # self.zivid.start()
        self.block = BlockDetection3D(self.Tpc)
        self.ball = BallDetectionRGBD(Trc=self.Trc, Tpc=self.Tpc, which_camera=which_camera)
        self.gp = GraspingPose3D(which_arm='PSM1')
        self.vd = VisualizeDetection()
        self.pegboard = PegboardCalibration()
        self.dvrk_motion = dvrkPegTransferMotion()
        self.dvrk_model = dvrkKinematics()

        # action ordering
        action_l2r = np.array([[1, 7], [3, 8], [5, 11], [0, 6], [2, 9], [4, 10]])
        action_r2l = action_l2r[:, [1, 0]]
        self.action_list = np.concatenate((action_l2r, action_r2l), axis=0)
        self.action_order = 0

        # data members
        self.color = []
        self.point = []
        self.state = ['initialize', None, None, None, None]
        self.main()

    def transform_task2robot(self, point, delta=False, inverse=False):
        point = np.array(point)
        if inverse == False:
            Tcp = np.linalg.inv(self.block.Tpc)  # (mm)
            Tcp[:3, -1] = Tcp[:3, -1] * 0.001  # (m)
            Trp = self.Trc.dot(Tcp)
            Rrp = Trp[:3, :3]
            trp = Trp[:3, -1]
            if delta == False:
                transformed = Rrp.dot(point * 0.001) + trp
            else:
                transformed = Rrp.dot(point * 0.001)
        else:
            Tcr = np.linalg.inv(self.Trc)   # (m)
            Tcr[:3,-1] = Tcr[:3,-1] * 1000  # (mm)
            Tpr = self.block.Tpc.dot(Tcr)
            Rpr = Tpr[:3, :3]
            tpr = Tpr[:3, -1]
            if delta == False:
                transformed = Rpr.dot(point * 1000) + tpr
            else:
                transformed = Rpr.dot(point * 1000)
        return transformed

    @classmethod
    def transform(cls, points, T):
        R = T[:3, :3]
        t = T[:3, -1]
        return R.dot(points.T).T + t.T

    def move_blocks(self):
        # [n, ang, x,y,z, seen]
        gp_pick = self.gp.pose_grasping
        gp_place = self.gp.pose_placing

        # pick-up motion
        gp_pick_robot = self.transform_task2robot(gp_pick[1:])  # [x,y,z]
        self.dvrk_motion.pick_above_block(pos1=gp_pick_robot, rot1=gp_pick[0])

        # visual servoing prior to pick-up
        # delta = self.pick_servoing()
        # delta_robot = self.transform_task2robot(delta, delta=True)
        delta_robot = np.array([0.0, 0.0, 0.0])
        self.dvrk_motion.pick_block(pos1=gp_pick_robot + delta_robot, rot1=gp_pick[0])

        # place motion
        gp_place_robot = self.transform_task2robot(gp_place[1:])
        self.dvrk_motion.place_above_block(pos1=gp_place_robot, rot1=gp_place[0])

        # visual servoing prior to drop
        # delta, seen = self.place_servoing()
        # if seen == True:    # if there is a block,
        # delta_robot = self.transform_task2robot(delta, delta=True)
        delta_robot = np.array([0.0, 0.0, 0.0])
        self.dvrk_motion.drop_block(pos1=gp_place_robot + delta_robot, rot1=gp_place[0])

    def pick_servoing(self):
        # get current wrist position from two spheres
        self.update_images()
        pbr = self.ball.find_balls(self.color, self.point, 'red')
        pt_act = self.ball.find_tool_position(pbr[0], pbr[1])
        pt_act = self.transform(pt_act, self.block.Tpc)     # w.r.t task coordinate

        # calculate desired wrist position from the desired grasping pose
        pos = self.gp.pose_grasping[1:]
        pos_robot = self.transform_task2robot(pos)
        rot = [self.gp.pose_grasping[0], 0.0, 0.0]
        quat = U.euler_to_quaternion(rot, 'deg')
        joints = self.dvrk_model.pose_to_joint(pos_robot, quat)
        pt_des_robot = dvrkKinematics.fk_position(joints, L1=dvrkVar.L1, L2=dvrkVar.L2)
        pt_des = self.transform_task2robot(pt_des_robot, inverse=True) + np.array([0.0, 0.0, -30])
        delta = pt_des - pt_act
        return delta

    def place_servoing(self):
        # get current position
        self.update_images()
        T, pnt_blk, pnt_mask = self.block.find_block_servo(self.color, self.point)
        if len(pnt_blk) < 400:
            delta = []
            seen = False
        else:
            # self.vd.plot3d(pnt_blocks=pnt_blk, pnt_masks=pnt_mask, pnt_pegs=self.block.pnt_pegs)
            nb_place = self.action_list[self.action_order][1]
            p_peg = self.block.pnt_pegs[nb_place] + np.array([0.0, 0.0, -5])  # sub-mm above peg
            p_fid = [0, 0, 15]
            p_fid = T[:3, :3].dot(p_fid) + T[:3,-1]     # w.r.t task coordinate
            delta = p_peg - p_fid
            seen = True
        return delta, seen

    def update_images(self):
        # self.color, _, self.point = self.zivid.capture_3Dimage(color='BGR')
        self.color = np.load('record/peg_transfer_kit_capture/img_color_inclined.npy')
        self.point = np.load('record/peg_transfer_kit_capture/img_point_inclined.npy')

    def main(self):
        while True:
            if self.state[0] == 'initialize':
                print('')
                print('* State:', self.state[0])
                # define ROI
                self.dvrk_motion.move_origin()

                # random move when NN model uses history
                # self.dvrk_motion.move_random()

                self.update_images()
                ycr, hcr, xcr, wcr = self.pegboard.define_boundary(self.color)
                dx = 200    # clearance of the region of interest
                dy = 200
                self.zivid.ycr = ycr - dy
                self.zivid.hcr = hcr + 2 * dy
                self.zivid.xcr = xcr - dx
                self.zivid.wcr = wcr + 2 * dx

                # find pegs
                self.block.find_pegs(img_color=self.color, img_point=self.point)
                self.state.insert(0, 'update_image')

            elif self.state[0] == 'update_image':
                print('')
                print('* State:', self.state[0])
                self.update_images()
                print ('image updated.')
                self.state.insert(0, 'plan_action')

            elif self.state[0] == 'plan_action':
                print('')
                print('* State:', self.state[0])
                if self.action_order == len(self.action_list):
                    self.state.insert(0, 'exit')
                else:
                    self.state.insert(0, 'find_block')
                    print("action pair #", self.action_list[self.action_order])

            elif self.state[0] == 'find_block':
                print('')
                print('* State:', self.state[0])
                nb_pick = self.action_list[self.action_order][0]
                nb_place = self.action_list[self.action_order][1]
                pose_blk_pick, pnt_blk_pick, pnt_mask_pick = self.block.find_block(
                    block_number=nb_pick, img_color=self.color, img_point=self.point)
                pose_blk_place, pnt_blk_place, pnt_mask_place = self.block.find_block(
                    block_number=nb_place, img_color=self.color, img_point=self.point)

                # check if there is block to move
                if pose_blk_pick != [] and pose_blk_place == []:
                    print('A block to move was detected.')
                    # find grasping & placing pose
                    self.gp.find_grasping_pose(pose_blk=pose_blk_pick, peg_point=self.block.pnt_pegs[nb_pick])
                    self.gp.find_placing_pose(peg_point=self.block.pnt_pegs[nb_place])
                    # visualize
                    # pnt_grasping = np.array(self.gp.pose_grasping)[1:]
                    # self.vd.plot3d(pnt_blk_pick, pnt_mask_pick, self.block.pnt_pegs, [pnt_grasping])
                    self.state.insert(0, 'move_block')
                else:
                    print('No block to move was detected. Skip this order.')
                    self.action_order += 1
                    self.state.insert(0, 'plan_action')

            elif self.state[0] == 'move_block':
                print('')
                print('* State:', self.state[0])
                self.move_blocks()
                self.dvrk_motion.move_origin()
                self.action_order += 1
                self.state.insert(0, 'update_image')

            elif self.state[0] == 'exit':
                print("task completed.")
                self.dvrk_motion.move_origin()
                exit()

if __name__ == '__main__':
    FLS = FLSPegTransfer(which_camera='inclined')