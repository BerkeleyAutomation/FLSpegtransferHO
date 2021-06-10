from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.ZividUtils import ZividUtils
from FLSpegtransfer.vision.GraspingPose3D import GraspingPose3D
from FLSpegtransfer.vision.VisualizeDetection import VisualizeDetection
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration
from FLSpegtransfer.motion.dvrkPegTransferMotionSingleArm import dvrkPegTransferMotionSingleArm
from FLSpegtransfer.utils.ImgUtils import ImgUtils
from FLSpegtransfer.path import *
import numpy as np
import time


class FLSPegTransfer:
    def __init__(self, use_simulation=True, use_controller=True, use_optimization=True, optimizer='cubic', which_camera='inclined', which_arm='PSM1'):
        self.use_simulation = use_simulation

        # load transform
        self.Trc = np.load(root + 'calibration_files/Trc_' + which_camera + '_'+which_arm+'.npy')  # robot to camera
        self.Tpc = np.load(root + 'calibration_files/Tpc_' + which_camera + '.npy')  # pegboard to camera

        # self.Trc_stereo = np.load(root + 'calibration_files/Trc_' + 'stereo' + '_'+which_arm+'.npy')  # robot to camera
        # self.Trc_stereo[:3, -1] += np.array([0.00, 0.003, 0.00])
        # self.Tc1c2 = np.linalg.inv(self.Trc).dot(self.Trc_stereo)    # cam1 = inclined, cam2 = stereo

        # import modules
        if self.use_simulation:
            pass
        else:
            self.zivid = ZividCapture(which_camera=which_camera)
            self.zivid.start()
        self.block = BlockDetection3D(self.Tpc)
        self.gp = GraspingPose3D(dist_gps=5, dist_pps=5, which_arm=which_arm)
        self.vd = VisualizeDetection()
        self.pegboard = PegboardCalibration()
        self.dvrk_motion\
            = dvrkPegTransferMotionSingleArm(use_controller=use_controller, use_optimization=use_optimization,
                                             optimizer=optimizer, which_arm=which_arm)
        self.zivid_utils = ZividUtils(which_camera='inclined')

        # action ordering
        self.action_list = np.array([[1, 7], [0, 6], [3, 8], [2, 9], [5, 11], [4, 10],  # left to right
                                     [7, 1], [6, 0], [8, 3], [9, 2], [11, 5], [10, 4]])  # right to left
        self.action_order = 0

        # data members
        self.color = []
        self.point = []
        self.state = ['initialize']
        self.main()

    def transform_task2robot(self, point, delta=False, inverse=False):
        point = np.array(point)
        if inverse == False:
            Tcp = np.linalg.inv(self.block.Tpc)
            Trp = self.Trc.dot(Tcp)
            Rrp = Trp[:3, :3]
            trp = Trp[:3, -1]
            if delta == False:
                transformed = Rrp.dot(point) + trp
            else:
                transformed = Rrp.dot(point)
        else:
            Tcr = np.linalg.inv(self.Trc)
            Tpr = self.block.Tpc.dot(Tcr)
            Rpr = Tpr[:3, :3]
            tpr = Tpr[:3, -1]
            if delta == False:
                transformed = Rpr.dot(point) + tpr
            else:
                transformed = Rpr.dot(point)
        return transformed

    @classmethod
    def transform(cls, points, T):
        R = T[:3, :3]
        t = T[:3, -1]
        return R.dot(points.T).T + t.T

    def move_blocks(self):
        # [n, ang, x,y,z, seen]
        gp_pick = self.gp.pose_grasping
        gp_place = np.array(self.gp.pose_placing)

        # pick-up and place motion
        gp_pick_robot = self.transform_task2robot(gp_pick[1:]*0.001)  # [x,y,z]
        gp_place_robot = self.transform_task2robot(gp_place[1:]*0.001)

        if self.action_order == 0 or self.action_order == 6:
            self.dvrk_motion.go_pick(pos_pick=gp_pick_robot, rot_pick=gp_pick[0])
        else:
            self.dvrk_motion.return_to_peg(pos_pick=gp_pick_robot, rot_pick=gp_pick[0])
        self.dvrk_motion.transfer_block(pos_pick=gp_pick_robot, rot_pick=gp_pick[0],
                                        pos_place=gp_place_robot, rot_place=gp_place[0])

        # visual correction
        delta, seen = self.place_servoing()
        delta_rob = self.transform_task2robot(delta, delta=True)
        delta_rob[2] = 0.0  # temporary
        self.dvrk_motion.servoing_block(pos_place=gp_place_robot+delta_rob, rot_place=gp_place[0])

    def place_servoing(self):
        # get current position
        self.update_images()
        nb_place = self.action_list[self.action_order][1]
        pose_blk, pnt_blk, pnt_mask = self.block.find_block_servo(self.color, self.point, self.block.pnt_pegs[nb_place])
        if len(pnt_blk) < 400:
            delta = [0.0, 0.0, 0.0]
            seen = False
        else:
            # self.vd.plot3d(pnt_blocks=[pnt_blk], pnt_masks=[pnt_mask], pnt_pegs=self.block.pnt_pegs)
            _, T = pose_blk
            p_peg = self.block.pnt_pegs[nb_place] + np.array([0.0, 0.0, -5])  # sub-mm above peg
            p_fid = [0, 0, 15]
            p_fid = T[:3, :3].dot(p_fid) + T[:3,-1]     # w.r.t task coordinate
            delta = (p_peg - p_fid)*0.001   # unit to (m)
            seen = True
        return delta, seen

    def update_images(self):
        if self.use_simulation:
            if self.action_order <= 5:
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
                print ("started")
                time.sleep(3)
                st = time.time()

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
                    # self.vd.plot3d([pnt_blk_pick], [pnt_mask_pick], self.block.pnt_pegs, [pnt_grasping])
                    # pnt_placing = np.array(self.gp.pose_placing)[1:]
                    # self.vd.plot3d([pnt_blk_place], [pnt_mask_pick], self.block.pnt_pegs, [pnt_placing])

                    # visualize all blocks and all grasping points
                    # self.block.find_block_all(img_color=self.color, img_point=self.point)
                    # self.gp.find_grasping_pose_all(pose_blks=self.block.pose_blks, peg_points=self.block.pnt_pegs)
                    # pnt_graspings = np.array(self.gp.pose_grasping)[:6, 2:5]
                    # self.vd.plot3d(self.block.pnt_blks, self.block.pnt_masks, self.block.pnt_pegs, pnt_graspings)
                    self.state.insert(0, 'move_block')
                else:
                    print('No block to move was detected. Skip this order.')
                    self.action_order += 1
                    self.state.insert(0, 'plan_action')

            elif self.state[0] == 'move_block':
                print('')
                print('* State:', self.state[0])
                self.move_blocks()
                if self.action_order == 5:
                    self.dvrk_motion.move_origin()
                self.action_order += 1
                self.state.insert(0, 'update_image')

            elif self.state[0] == 'exit':
                print("task completed.")
                self.dvrk_motion.move_origin()
                t_comp = time.time() - st
                print (t_comp)
                np.save('/home/hwangmh/t_comp', t_comp)

                print(self.dvrk_motion.time_motion)
                print (np.sum(self.dvrk_motion.time_motion))
                print(self.dvrk_motion.time_computing)
                print (np.sum(self.dvrk_motion.time_computing))
                import pdb; pdb.set_trace()
                exit()


if __name__ == '__main__':
    FLS = FLSPegTransfer(use_simulation=True, use_controller=False, use_optimization=True, optimizer='mtsqp', which_camera='inclined', which_arm='PSM1')