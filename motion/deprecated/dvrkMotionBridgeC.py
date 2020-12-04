from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.motion.deprecated.dvrkMotionBridgeP import dvrkMotionBridgeP
from FLSpegtransfer.trash.dvrkMotionControllerDNN import dvrkNNController


class dvrkMotionBridgeC(dvrkMotionBridgeP):
    def __init__(self):
        super(dvrkMotionBridgeC, self).__init__()
        root = "/home/hwangmh/pycharmprojects/FLSpegtransfer/"
        dir = "calibration_files/model_grey_history_peg_sampled/"
        # self.NNController = dvrkNNController(root+dir+"model_hysteresis.out", nb_ensemble=10)
        self.NNController = dvrkNNController(root + dir + "model_grey_history_peg_sampled.out", nb_ensemble=10)
        self.dvrk_model = dvrkKinematics()

    def set_pose(self, pos1=[], rot1=[], jaw1=[], pos2=[], rot2=[], jaw2=[]):
        joint1 = self.dvrk_model.pose_to_joint(pos1, rot1)
        self.set_joint(joint1=joint1, jaw1=jaw1, joint2=[], jaw2=[], use_interpolation=False)

    def set_joint(self, joint1=[], jaw1=[], joint2=[], jaw2=[], use_interpolation=False):
        if joint1 == []:
            joint1_cal = []
        else:
            if use_interpolation:
                joint1_cal = self.NNController.cal_interpolate(joint1, mode='calibrate')
            else:
                joint1_cal = self.NNController.step(joint1)
        super().set_joint(joint1=joint1_cal, jaw1=jaw1, joint2=joint2, jaw2=jaw2)