from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.GraspingPose3D import GraspingPose3D
from FLSpegtransfer.vision.VisualizeDetection import VisualizeDetection
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration
from FLSpegtransfer.motion.dvrkPegTransferMotionHandOver import dvrkPegTransferMotionHandOver
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.path import *
import threading
import numpy as np


class FLSPegTransferHandover:
    def __init__(self, use_simulation=True, use_controller=True, use_optimization=True, optimizer='cubic', which_camera='inclined'):
        self.use_simulation = use_simulation

        # load transform
        self.Trc1 = np.load(root + 'calibration_files/Trc_' + which_camera + '_PSM1.npy')  # robot to camera
        self.Trc2 = np.load(root + 'calibration_files/Trc_' + which_camera + '_PSM2.npy')  # robot to camera
        self.Trc2[:3, -1] += np.array([-0.001, 0.00, 0.00])
        self.Tpc = np.load(root + 'calibration_files/Tpc_' + which_camera + '.npy')  # pegboard to camera
        # import modules
        if use_simulation:
            pass
        else:
            self.zivid = ZividCapture(which_camera=which_camera)
            self.zivid.start()
        self.block = BlockDetection3D(self.Tpc)
        self.gp = {'PSM1': GraspingPose3D(dist_pps=5, dist_gps=5, which_arm='PSM1'),
                   'PSM2': GraspingPose3D(dist_pps=5, dist_gps=5, which_arm='PSM2')}
        self.vd = VisualizeDetection()
        self.pegboard = PegboardCalibration()
        self.dvrk_motion\
            = dvrkPegTransferMotionHandOver(use_controller=use_controller, use_optimization=use_optimization, optimizer=optimizer)
        self.dvrk_model = dvrkKinematics()

        # action ordering (pick, place) pair
        self.action_list = [['PSM2', 1, 'PSM1', 7],     # left to right
                            ['PSM2', 0, 'PSM1', 6],
                            ['PSM2', 3, 'PSM1', 8],
                            ['PSM2', 2, 'PSM1', 9],
                            ['PSM2', 5, 'PSM1', 11],
                            ['PSM2', 4, 'PSM1', 10],
                            ['PSM1', 7, 'PSM2', 1],     # right to left
                            ['PSM1', 6, 'PSM2', 0],
                            ['PSM1', 8, 'PSM2', 3],
                            ['PSM1', 9, 'PSM2', 2],
                            ['PSM1', 11, 'PSM2', 5],
                            ['PSM1', 10, 'PSM2', 4]]
        self.action_order = 0

        # data members
        self.color = []
        self.point = []
        self.state = ['initialize', 'initialize']   # [pick state, place state]

        # event
        self.event_handover = threading.Event()
        self.event_waiting = threading.Event()
        self.event_blockready = threading.Event()

        # parallelize
        self.initialize()
        self.thread1 = threading.Thread(target=self.run_pick)
        self.thread2 = threading.Thread(target=self.run_place)
        self.thread1.start()
        self.thread2.start()
        # self.run_place()

    def initialize(self):
        self.dvrk_motion.move_origin()
        # self.dvrk_motion.move_random(which_arm='PSM1')    # random move when NN model uses history
        self.update_images()
        self.block.find_pegs(img_color=self.color, img_point=self.point)
        self.pnt_handover = (self.block.pnt_pegs[3] + self.block.pnt_pegs[6] + self.block.pnt_pegs[9]) / 3 + np.array([0.0, -10.0, 0.0])
        self.state = ['plan_action', 'plan_action']

    def transform_task2robot(self, point, delta=False, inverse=False, which_arm='PSM1'):
        point = np.array(point)
        if inverse == False:
            Tcp = np.linalg.inv(self.Tpc)
            if which_arm=='PSM1' or which_arm==0:
                Trp = self.Trc1.dot(Tcp)
            elif which_arm=='PSM2' or which_arm==1:
                Trp = self.Trc2.dot(Tcp)
            else:
                raise ValueError
            Rrp = Trp[:3, :3]
            trp = Trp[:3, -1]
            if delta == False:
                transformed = Rrp.dot(point) + trp
            else:
                transformed = Rrp.dot(point)
        else:
            if which_arm=='PSM1':
                Tcr = np.linalg.inv(self.Trc1)
            elif which_arm=='PSM2':
                Tcr = np.linalg.inv(self.Trc2)
            else:
                raise ValueError
            Tpr = self.Tpc.dot(Tcr)
            Rpr = Tpr[:3, :3]
            tpr = Tpr[:3, -1]
            if delta == False:
                transformed = Rpr.dot(point) + tpr
            else:
                transformed = Rpr.dot(point)
        return transformed

    def place_servoing(self, nb_place_corr):
        # get current position
        self.update_images()
        pose_blk, pnt_blk, pnt_mask = self.block.find_block_servo(self.color, self.point, self.block.pnt_pegs[nb_place_corr])
        if len(pnt_blk) < 400:
            delta = [0.0, 0.0, 0.0]
            seen = False
        else:
            # self.vd.plot3d(pnt_blocks=[pnt_blk], pnt_masks=[pnt_mask], pnt_pegs=self.block.pnt_pegs)
            _, T = pose_blk
            p_peg = self.block.pnt_pegs[nb_place_corr] + np.array([0.0, 0.0, -5])  # sub-mm above peg
            p_fid = [0, 0, 15]
            p_fid = T[:3, :3].dot(p_fid) + T[:3, -1]  # w.r.t task coordinate
            delta = (p_peg - p_fid) * 0.001  # unit to (m)
            seen = True
        return delta, seen

    def update_images(self):
        if self.use_simulation:
            if self.action_order <= 5:
                self.color = np.load('record/peg_transfer_kit_capture/img_color_inclined_.npy')
                self.point = np.load('record/peg_transfer_kit_capture/img_point_inclined_.npy')
                # self.color = np.load('record/peg_transfer_kit_capture/img_color_inclined_rhp.npy')
                # self.point = np.load('record/peg_transfer_kit_capture/img_point_inclined_rhp.npy')
            else:
                self.color = np.load('record/peg_transfer_kit_capture/img_color_inclined_.npy')
                self.point = np.load('record/peg_transfer_kit_capture/img_point_inclined_.npy')
                # self.color = np.load('record/peg_transfer_kit_capture/img_color_inclined_rhp.npy')
                # self.point = np.load('record/peg_transfer_kit_capture/img_point_inclined_rhp.npy')
        else:
            self.color, _, self.point = self.zivid.capture_3Dimage(color='BGR')

    def run_pick(self):
        while True:
            if self.state[0] == 'plan_action':
                if self.action_order == len(self.action_list):
                    print ("Pick Task Completed!")
                    exit()
                elif self.action_order == 6:
                    self.event_blockready.set()
                    self.event_blockready.clear()
                    self.event_waiting.clear()
                    self.event_waiting.wait()
                print('')
                print("Action order: ", self.action_order, "started")
                self.state[0] = 'find_block'

            elif self.state[0] == 'find_block':
                # find block
                self.update_images()
                nb_pick = self.action_list[self.action_order][1]
                arm_pick = self.action_list[self.action_order][0]
                pose_blk_pick, pnt_blk_pick, pnt_mask_pick = self.block.find_block(
                    block_number=nb_pick, img_color=self.color, img_point=self.point)

                # check if there is block to move
                if pose_blk_pick != []:
                    # find grasping & placing pose
                    self.gp[arm_pick].find_grasping_pose(pose_blk=pose_blk_pick, peg_point=self.block.pnt_pegs[nb_pick])
                    self.gp[arm_pick].find_placing_pose(peg_point=self.pnt_handover)

                    # # visualize
                    # pnt_grasping = np.array(self.gp[arm_pick].pose_grasping)[1:]
                    # self.vd.plot3d([pnt_blk_pick], [pnt_mask_pick], self.block.pnt_pegs, [pnt_grasping])

                    self.state[0] = 'move_block'
                else:  # no block detected
                    print ("No block detected for pick-up, Skip this order.")
                    self.action_order += 1
                    self.state[0] = 'plan_action'

            elif self.state[0] == 'move_block':
                # print(' ')
                # print("action order ", self.action_order)
                # print("action pair #", self.action_list[self.action_order[0]])
                # print(' ')
                arm_pick = self.action_list[self.action_order][0]
                gp_pick = self.gp[arm_pick].pose_grasping      # [n, ang, x,y,z, seen]
                gp_place = self.gp[arm_pick].pose_placing      # [n, ang, x,y,z, seen]
                gp_place = np.array(gp_place)
                gp_pick_robot = self.transform_task2robot(gp_pick[1:]*0.001, which_arm=arm_pick)  # [x,y,z]
                gp_place_robot = self.transform_task2robot(gp_place[1:]*0.001, which_arm=arm_pick)  # [x,y,z]

                # pick up block and move to center
                self.dvrk_motion.go_pick(pos_pick=gp_pick_robot, rot_pick=gp_pick[0], which_arm=arm_pick)
                self.dvrk_motion.go_handover(pos_pick=gp_pick_robot, rot_pick=gp_pick[0], pos_place=gp_place_robot, rot_place=gp_place[0], which_arm=arm_pick)
                self.event_blockready.set()
                self.event_handover.wait()   # until block is grasped by other gripper
                self.event_handover.clear()

                # escape from hand-over
                self.dvrk_motion.move_jaw(jaw='open', which_arm=arm_pick)
                self.dvrk_motion.move_upright(pos=[gp_place_robot[0], gp_place_robot[1], self.dvrk_motion.height_handover+self.dvrk_motion.offset_handover],
                                              rot=gp_place[0], jaw='open', which_arm=arm_pick)
                self.event_handover.set()
                self.action_order += 1
                self.state[0] = 'plan_action'

    def run_place(self):
        while True:
            if self.state[1] == 'plan_action':
                if self.action_order == len(self.action_list):
                    self.dvrk_motion.move_origin()
                    print ("Place Task Completed!")
                    exit()
                elif self.action_order == 6:
                    self.dvrk_motion.move_origin()
                    self.event_waiting.set()
                self.state[1] = 'find_block'

            elif self.state[1] == 'find_block':
                arm_place = self.action_list[self.action_order][2]
                self.dvrk_motion.handover_ready(which_arm=arm_place)
                self.event_blockready.wait()  # until block is ready to hand-over
                self.event_blockready.clear()
                if self.use_simulation:
                    self.color = np.load('record/peg_transfer_kit_capture/img_color_inclined_ho.npy')
                    self.point = np.load('record/peg_transfer_kit_capture/img_point_inclined_ho.npy')
                else:
                    self.update_images()

                nb_place = self.action_list[self.action_order][3]
                arm_place = self.action_list[self.action_order][2]
                nb_place_corr = nb_place
                pose_blk_pick, pnt_blk_pick, pnt_mask_pick = self.block.find_block_servo_handover(
                    img_color=self.color, img_point=self.point)

                # check if there is block to move
                if pose_blk_pick != []:
                    # find grasping & placing pose
                    self.gp[arm_place].find_grasping_pose_handover(pose_blk=pose_blk_pick, which_arm=arm_place)
                    self.gp[arm_place].find_placing_pose(peg_point=self.block.pnt_pegs[nb_place])

                    # visualize
                    # pnt_grasping = np.array(self.gp[arm_place].pose_grasping)[1:]
                    # np.save('pnt_blk_pick', pnt_blk_pick)
                    # np.save('pnt_mask_pick', pnt_mask_pick)
                    # np.save('block_pnt_pegs', self.block.pnt_pegs)
                    # np.save('pnt_grasping', pnt_grasping)
                    # self.vd.plot3d([pnt_blk_pick], [pnt_mask_pick], self.block.pnt_pegs, [pnt_grasping])
                    self.state[1] = 'move_block'
                else:
                    print("No block detected for hand-over, Skip this order.")
                    self.event_handover.set()
                    self.state[1] = 'plan_action'

            elif self.state[1] == 'move_block':
                # [n, ang, x,y,z, seen]
                # nb_place = self.action_list[self.action_order][3]
                arm_place = self.action_list[self.action_order][2]
                gp_pick = self.gp[arm_place].pose_grasping
                gp_place = self.gp[arm_place].pose_placing
                gp_place = np.array(gp_place)
                gp_pick_robot = self.transform_task2robot(gp_pick[1:]*0.001, which_arm=arm_place)  # [x,y,z]
                gp_place_robot = self.transform_task2robot(gp_place[1:]*0.001, which_arm=arm_place)  # [x,y,z]

                # pick (hand-over)
                self.dvrk_motion.pick_block(pos=gp_pick_robot, rot=gp_pick[0], which_arm=arm_place)
                self.event_handover.set()
                self.event_handover.clear()
                self.event_handover.wait()   # until other gripper escape
                self.event_handover.clear()

                # place
                self.dvrk_motion.go_place(pos_place=gp_place_robot, rot_place=gp_place[0], which_arm=arm_place)

                # visual correction
                delta, seen = self.place_servoing(nb_place_corr)
                delta_rob = self.transform_task2robot(delta, delta=True)
                delta_rob[2] = 0.0  # temporary
                print(delta_rob)
                self.dvrk_motion.servoing_block(pos_place=gp_place_robot + delta_rob, rot_place=gp_place[0], which_arm=arm_place)
                print('')
                print("Action completed")
                self.state[1] = 'plan_action'


if __name__ == '__main__':

    FLS = FLSPegTransferHandover(use_simulation=False, use_controller=False, use_optimization=False, optimizer='qp', which_camera='inclined')