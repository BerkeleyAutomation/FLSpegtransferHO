from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.GraspingPose3D import GraspingPose3D
from FLSpegtransfer.vision.VisualizeDetection import VisualizeDetection
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration
from FLSpegtransfer.motion.dvrkPegTransferMotionHandOver import dvrkPegTransferMotionHandOver
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.path import *
import threading, time
import numpy as np


class FLSPegTransfer():
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
        # self.gp = [GraspingPose3D(which_arm='PSM1'), GraspingPose3D(which_arm='PSM2')]
        self.gp = {'PSM1':GraspingPose3D(which_arm='PSM1'), 'PSM2':GraspingPose3D(which_arm='PSM2')}
        self.vd = VisualizeDetection()
        self.pegboard = PegboardCalibration()
        self.dvrk_motion = dvrkPegTransferMotionHandOver()
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
        self.action_order = [-1, -1]  # [pick action, place action]

        # data members
        self.color = []
        self.point = []
        self.state = []   # [pick state, place state]
        self.pnt_handover = []

        # event
        self.event_handover = threading.Event()
        self.event_waiting = threading.Event()

        # parallelize
        self.initialize()

        # self.thread1 = threading.Thread(target=self.PSM1)
        self.thread2 = threading.Thread(target=self.run_pick)
        # self.thread1.start()
        self.thread2.start()
        self.run_place()

    def initialize(self):
        self.dvrk_motion.move_origin()
        # self.dvrk_motion.move_random(which_arm='PSM1')    # random move when NN model uses history
        self.update_images()
        self.block.find_pegs(img_color=self.color, img_point=self.point)
        self.pnt_handover = (self.block.pnt_pegs[3] + self.block.pnt_pegs[6] + self.block.pnt_pegs[9]) / 3
        self.state = ['plan_action', 'plan_action']

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

    def place_servoing(self, nb_place):
        # get current position
        self.update_images()
        (_, T), pnt_blk, pnt_mask = self.block.find_block_servo(self.color, self.point)
        if len(pnt_blk) < 400:
            delta = []
            seen = False
        else:
            # self.vd.plot3d(pnt_blocks=pnt_blk, pnt_masks=pnt_mask, pnt_pegs=self.block.pnt_pegs)
            p_peg = self.block.pnt_pegs[nb_place] + np.array([0.0, 0.0, -5])  # sub-mm above peg
            p_fid = [0, 0, 15]
            p_fid = T[:3, :3].dot(p_fid) + T[:3,-1]     # w.r.t task coordinate
            delta = p_peg - p_fid
            seen = True
        return delta, seen

    def update_images(self):
        if self.use_simulation:
            if self.action_order[0] <= 6 and self.action_order[1] <= 5:
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
            time.sleep(0.2)
            if self.state[0] == 'plan_action' or self.state[0] == 'no_block':
                if self.action_order[0] == len(self.action_list)-1:
                    print ("Pick Task Completed!")
                    exit()
                else:
                    if self.state[0] == 'no_block':
                        print ("No block detected for pick-up, Skip this order.")
                        self.action_order[1] += 1
                    arm_pick = self.action_list[self.action_order[0]][0]
                    arm_pick_next = self.action_list[self.action_order[0]+1][0]
                    print('')
                    print("("+arm_pick+")", " Pick order: ", self.action_order[0], "th completed")
                    if arm_pick != arm_pick_next:
                        self.event_waiting.clear()
                        self.event_waiting.wait()
                    self.action_order[0] += 1
                    self.state[0] = 'find_block'

            elif self.state[0] == 'wait':
                if self.state[1] == 'picked_hand_over':
                    self.state[0] = 'hand_off'
                    self.state[1] = 'wait'

            elif self.state[0] == 'find_block':
                # find block
                self.update_images()
                nb_pick = self.action_list[self.action_order[0]][1]
                arm_pick = self.action_list[self.action_order[0]][0]
                pose_blk_pick, pnt_blk_pick, pnt_mask_pick = self.block.find_block(
                    block_number=nb_pick, img_color=self.color, img_point=self.point)

                # check if there is block to move
                if pose_blk_pick != []:
                    # find grasping & placing pose
                    self.gp[arm_pick].find_grasping_pose(pose_blk=pose_blk_pick, peg_point=self.block.pnt_pegs[nb_pick])
                    self.gp[arm_pick].find_placing_pose(peg_point=self.pnt_handover)
                    self.state[0] = 'move_block'
                else:  # no block detected
                    self.state[0] = 'no_block'

            elif self.state[0] == 'move_block':
                # print(' ')
                # print("action order ", self.action_order)
                # print("action pair #", self.action_list[self.action_order[0]])
                # print(' ')
                arm_pick = self.action_list[self.action_order[0]][0]
                gp_pick = self.gp[arm_pick].pose_grasping      # [n, ang, x,y,z, seen]
                gp_place = self.gp[arm_pick].pose_placing      # [n, ang, x,y,z, seen]
                gp_pick_robot = self.transform_task2robot(gp_pick[1:], which_arm=arm_pick)  # [x,y,z]
                gp_place_robot = self.transform_task2robot(gp_place[1:], which_arm=arm_pick)  # [x,y,z]

                # pick up block and move to center
                self.dvrk_motion.move_upright(pos=[gp_pick_robot[0], gp_pick_robot[1], self.dvrk_motion.height_ready],
                                              rot=gp_pick[0], jaw='close', which_arm=arm_pick)
                self.dvrk_motion.pick_block(pos=gp_pick_robot, rot=gp_pick[0], which_arm=arm_pick)
                self.dvrk_motion.move_upright(pos=[gp_pick_robot[0], gp_pick_robot[1], self.dvrk_motion.height_ready],
                                              rot=gp_pick[0], jaw='close', which_arm=arm_pick, interpolate=True)
                self.dvrk_motion.move_upright(pos=[gp_place_robot[0], gp_place_robot[1], self.dvrk_motion.height_ready],
                                              rot=gp_place[0], jaw='close', which_arm=arm_pick)
                self.state[0] = 'ready_hand_over'
            elif self.state[0] == 'hand_off':   # escape from hand-over
                arm_pick = self.action_list[self.action_order[0]][0]
                gp_place = self.gp[arm_pick].pose_placing  # [n, ang, x,y,z, seen]
                gp_place_robot = self.transform_task2robot(gp_place[1:], which_arm=arm_pick)  # [x,y,z]
                self.dvrk_motion.move_jaw(jaw='open', which_arm=arm_pick)
                self.dvrk_motion.move_upright(pos=[gp_place_robot[0], gp_place_robot[1], self.dvrk_motion.height_ready+self.dvrk_motion.offset_handover],
                                              rot=gp_place[0], jaw='open', which_arm=arm_pick)
                self.state[1] = 'place_block'
                self.state[0] = 'plan_action'

    def run_place(self):
        while True:
            time.sleep(0.1)
            if self.state[1] == 'plan_action' or self.state[1] == 'no_block':
                if self.action_order[1] == len(self.action_list)-1:
                    self.dvrk_motion.move_origin()
                    print ("Place Task Completed!")
                    exit()
                else:
                    if self.state[1] == 'no_block':
                        print("No block detected for hand-over, Skip this order.")
                        self.event_handover.set()
                    arm_place = self.action_list[self.action_order[1]][2]
                    arm_place_next = self.action_list[self.action_order[1]+1][2]
                    print('')
                    print("(" + arm_place + ")", " Place order: ", self.action_order[1], "th completed")
                    if arm_place != arm_place_next:
                        self.dvrk_motion.move_origin()
                        self.event_waiting.set()
                        self.action_order[1] = self.action_order[0]
                    self.action_order[1] += 1
                    self.state[1] = 'wait'

            elif self.state[1] == 'wait':
                if self.state[0] == 'ready_hand_over':
                    self.state[1] = 'find_block'
                    self.state[0] = 'wait'
                elif self.state[0] == 'hand_off':
                    self.state[1] = 'place_block'

            elif self.state[1] == 'find_block':
                # find block
                if self.use_simulation:
                    self.color = np.load('record/peg_transfer_kit_capture/img_color_inclined_ho.npy')
                    self.point = np.load('record/peg_transfer_kit_capture/img_point_inclined_ho.npy')
                    # self.color = np.load('record/peg_transfer_kit_capture/img_color_inclined.npy')
                    # self.point = np.load('record/peg_transfer_kit_capture/img_point_inclined.npy')
                else:
                    self.update_images()
                nb_place = self.action_list[self.action_order[1]][3]
                arm_place = self.action_list[self.action_order[1]][2]
                pose_blk_pick, pnt_blk_pick, pnt_mask_pick = self.block.find_block_servo(
                    img_color=self.color, img_point=self.point)
                # pose_blk_place, pnt_blk_place, pnt_mask_place = self.block.find_block(
                #     block_number=nb_place, img_color=self.color, img_point=self.point)

                # check if there is block to move
                if pose_blk_pick != []:
                    # find grasping & placing pose
                    self.gp[arm_place].find_grasping_pose_handover(pose_blk=pose_blk_pick)
                    self.gp[arm_place].find_placing_pose(peg_point=self.block.pnt_pegs[nb_place])
                    # visualize
                    # pnt_grasping = np.array(self.gp[0].pose_grasping)[1:]
                    # self.vd.plot3d(pnt_blk_pick, pnt_mask_pick, self.block.pnt_pegs, [pnt_grasping])
                    self.state[1] = 'move_block'
                else:
                    self.state[1] = 'no_block'

            elif self.state[1] == 'move_block':
                # [n, ang, x,y,z, seen]
                arm_place = self.action_list[self.action_order[1]][2]
                gp_pick = self.gp[arm_place].pose_grasping
                gp_pick_robot = self.transform_task2robot(gp_pick[1:], which_arm=arm_place)  # [x,y,z]

                # pick (hand-over)
                self.dvrk_motion.move_upright(pos=[gp_pick_robot[0], gp_pick_robot[1], gp_pick_robot[2]+self.dvrk_motion.offset_handover],
                                              rot=gp_pick[0], jaw='open', which_arm=arm_place)
                self.dvrk_motion.pick_block(pos=gp_pick_robot, rot=gp_pick[0], which_arm=arm_place)
                self.state[1] = 'picked_hand_over'

            elif self.state[1] == 'place_block':
                # place
                nb_place = self.action_list[self.action_order[1]][3]
                arm_place = self.action_list[self.action_order[1]][2]
                gp_place = self.gp[arm_place].pose_placing
                gp_place_robot = self.transform_task2robot(gp_place[1:], which_arm=arm_place)  # [x,y,z]
                pnt_handover_robot = self.transform_task2robot(self.pnt_handover, which_arm=arm_place)  # [x,y,z]
                self.dvrk_motion.move_upright(pos=[gp_place_robot[0], gp_place_robot[1], self.dvrk_motion.height_ready],
                                              rot=gp_place[0], jaw='close', which_arm=arm_place)
                # visual servoing prior to drop
                delta, seen = self.place_servoing(nb_place)
                if seen == True:    # if there is a block,
                    delta_robot = self.transform_task2robot(delta, delta=True)
                else:
                    delta_robot = [0.0, 0.0, 0.0]
                self.dvrk_motion.drop_block(pos=gp_place_robot+delta_robot, rot=gp_place[0], which_arm=arm_place)
                if arm_place == 'PSM1':     # at the first half
                    self.dvrk_motion.move_upright(pos=[pnt_handover_robot[0]-0.02, pnt_handover_robot[1], self.dvrk_motion.height_ready_handover],
                                                  rot=0.0, jaw='close', which_arm=arm_place)
                else:   # at the last half
                    self.dvrk_motion.move_upright(pos=[pnt_handover_robot[0]+0.02, pnt_handover_robot[1], self.dvrk_motion.height_ready_handover],
                                                  rot=0.0, jaw='close', which_arm=arm_place)
                self.state[1] = 'plan_action'


if __name__ == '__main__':
    FLS = FLSPegTransfer(use_simulation=True, which_camera='inclined')