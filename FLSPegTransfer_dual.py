from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.BallDetectionRGBD import BallDetectionRGBD
from FLSpegtransfer.vision.GraspingPose3D import GraspingPose3D
from FLSpegtransfer.vision.VisualizeDetection import VisualizeDetection
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration
from FLSpegtransfer.motion.dvrkPegTransferMotionDualArm import dvrkPegTransferMotionDualArm
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.path import *
import numpy as np


class FLSPegTransfer:
    def __init__(self, use_simulation=True, which_camera='inclined'):
        self.use_simulation = use_simulation

        # load transform
        self.Trc1 = np.load(root + 'calibration_files/Trc_' + which_camera + '_PSM1.npy')  # robot to camera
        self.Trc2 = np.load(root + 'calibration_files/Trc_' + which_camera + '_PSM2.npy')  # robot to camera
        self.Tpc = np.load(root + 'calibration_files/Tpc_' + which_camera + '.npy')  # pegboard to camera

        # import modules
        if use_simulation:
            pass
        else:
            self.zivid = ZividCapture(which_camera=which_camera)
            self.zivid.start()
        self.block = BlockDetection3D(self.Tpc)
        self.gp = {'PSM1': GraspingPose3D(which_arm='PSM1'), 'PSM2': GraspingPose3D(which_arm='PSM2')}
        self.vd = VisualizeDetection()
        self.pegboard = PegboardCalibration()
        self.dvrk_motion = dvrkPegTransferMotionDualArm()
        self.dvrk_model = dvrkKinematics()

        # action ordering
        self.action_list = np.array([[[0, 1], [7, 8]],
                                     [[2, 3], [6, 11]],
                                     [[4, 5], [9, 10]],  # left to right
                                     [[7, 8], [0, 1]],
                                     [[6, 11], [2, 3]],
                                     [[9, 10], [4, 5]]])  # right to left
        self.action_order = 0

        # data members
        self.color = []
        self.point = []
        self.state = ['initialize']
        self.main()

    def transform_task2robot(self, point, delta=False, inverse=False, which_arm='PSM1'):
        point = np.array(point)
        if inverse == False:
            Tcp = np.linalg.inv(self.Tpc)  # (mm)
            Tcp[:3, -1] = Tcp[:3, -1] * 0.001  # (m)
            if which_arm=='PSM1' or which_arm==0:
                Trp = self.Trc1.dot(Tcp)
            elif which_arm=='PSM2' or which_arm==1:
                Trp = self.Trc2.dot(Tcp)
            else:
                raise ValueError
            Rrp = Trp[:3, :3]
            trp = Trp[:3, -1]
            if delta == False:
                transformed = Rrp.dot(point * 0.001) + trp
            else:
                transformed = Rrp.dot(point * 0.001)
        else:
            if which_arm=='PSM1':
                Tcr = np.linalg.inv(self.Trc1)   # (m)
            elif which_arm=='PSM2':
                Tcr = np.linalg.inv(self.Trc2)   # (m)
            else:
                raise ValueError
            Tcr[:3,-1] = Tcr[:3,-1] * 1000  # (mm)
            Tpr = self.Tpc.dot(Tcr)
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
        gp_pick1 = self.gp['PSM1'].pose_grasping
        gp_place1 = self.gp['PSM1'].pose_placing
        gp_pick2 = self.gp['PSM2'].pose_grasping
        gp_place2 = self.gp['PSM2'].pose_placing

        # pick-up motion
        gp_pick_robot1 = self.transform_task2robot(gp_pick1[1:], which_arm='PSM1')  # [x,y,z]
        gp_pick_robot2 = self.transform_task2robot(gp_pick2[1:], which_arm='PSM2')
        print(gp_pick_robot1, gp_pick_robot2, gp_pick1[0], gp_pick2[0])
        print("pick_above_block")
        self.dvrk_motion.move_above_block(pos1=gp_pick_robot1, rot1=gp_pick1[0], pos2=gp_pick_robot2, rot2=gp_pick2[0])
        print("pick_block")
        self.dvrk_motion.pick_block(pos1=gp_pick_robot1, rot1=gp_pick1[0], pos2=gp_pick_robot2, rot2=gp_pick2[0])

        # place motion
        gp_place_robot1 = self.transform_task2robot(gp_place1[1:], which_arm='PSM1')
        gp_place_robot2 = self.transform_task2robot(gp_place2[1:], which_arm='PSM2')
        print(gp_place_robot1, gp_place_robot2, gp_place1[0], gp_place2[0])
        print("place_above_block")
        self.dvrk_motion.move_above_block(pos1=gp_place_robot1, rot1=gp_place1[0], pos2=gp_place_robot2, rot2=gp_place2[0])

        # visual servoing prior to drop
        # delta, seen = self.place_servoing()
        # if seen == True:    # if there is a block,
        #     delta_robot = self.transform_task2robot(delta, delta=True)
        self.dvrk_motion.drop_block(pos1=gp_place_robot1, rot1=gp_place1[0], pos2=gp_place_robot2, rot2=gp_place2[0])

    def place_servoing(self):
        # get current position
        self.update_images()
        pose_blk, pnt_blk, pnt_mask = self.block.find_block_servo(self.color, self.point)
        if len(pnt_blk) < 400:
            delta = []
            seen = False
        else:
            # self.vd.plot3d(pnt_blocks=pnt_blk, pnt_masks=pnt_mask, pnt_pegs=self.block.pnt_pegs)
            _, T = pose_blk
            nb_place = self.action_list[self.action_order][1]
            p_peg = self.block.pnt_pegs[nb_place] + np.array([0.0, 0.0, -5])  # sub-mm above peg
            p_fid = [0, 0, 15]
            p_fid = T[:3, :3].dot(p_fid) + T[:3,-1]     # w.r.t task coordinate
            delta = p_peg - p_fid
            seen = True
        return delta, seen

    def update_images(self):
        if self.use_simulation:
            if self.action_order <= 2:
                self.color = np.load('record/peg_transfer_kit_capture/img_color_inclined.npy')
                self.point = np.load('record/peg_transfer_kit_capture/img_point_inclined.npy')
            else:
                self.color = np.load('record/peg_transfer_kit_capture/img_color_inclined_rhp.npy')
                self.point = np.load('record/peg_transfer_kit_capture/img_point_inclined_rhp.npy')
        else:
            self.color, _, self.point = self.zivid.capture_3Dimage(color='BGR')

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
                nb_pick1 = self.action_list[self.action_order][0][1]    # PSM1
                nb_pick2 = self.action_list[self.action_order][0][0]    # PSM2
                nb_place1 = self.action_list[self.action_order][1][1]   # PSM1
                nb_place2 = self.action_list[self.action_order][1][0]   # PSM2
                pose_blk_pick1, pnt_blk_pick1, pnt_mask_pick1 = self.block.find_block(
                    block_number=nb_pick1, img_color=self.color, img_point=self.point)
                pose_blk_place1, pnt_blk_place1, pnt_mask_place1 = self.block.find_block(
                    block_number=nb_place1, img_color=self.color, img_point=self.point)
                pose_blk_pick2, pnt_blk_pick2, pnt_mask_pick2 = self.block.find_block(
                    block_number=nb_pick2, img_color=self.color, img_point=self.point)
                pose_blk_place2, pnt_blk_place2, pnt_mask_place2 = self.block.find_block(
                    block_number=nb_place2, img_color=self.color, img_point=self.point)

                # check if there is block to move
                if pose_blk_pick1 != [] and pose_blk_place1 == []\
                        and pose_blk_pick2 != [] and pose_blk_place2 == []:
                    print('A block to move was detected.')
                    # find grasping & placing pose
                    self.gp['PSM1'].find_grasping_pose(pose_blk=pose_blk_pick1, peg_point=self.block.pnt_pegs[nb_pick1])
                    self.gp['PSM1'].find_placing_pose(peg_point=self.block.pnt_pegs[nb_place1])
                    self.gp['PSM2'].find_grasping_pose(pose_blk=pose_blk_pick2, peg_point=self.block.pnt_pegs[nb_pick2])
                    self.gp['PSM2'].find_placing_pose(peg_point=self.block.pnt_pegs[nb_place2])
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
                self.action_order += 1
                if self.action_order == 3:
                    self.dvrk_motion.move_origin()
                self.state.insert(0, 'update_image')

            elif self.state[0] == 'exit':
                print("task completed.")
                self.dvrk_motion.move_origin()
                exit()

if __name__ == '__main__':
    FLS = FLSPegTransfer(use_simulation=True, which_camera='inclined')