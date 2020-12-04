from FLSpegtransfer.motion.dvrkArm import dvrkArm
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import FLSpegtransfer.utils.CmnUtil as U
import numpy as np


class dvrkDualArm(object):
    def __init__(self):
        self.arm1 = dvrkArm('/PSM1')
        self.arm2 = dvrkArm('/PSM2')
        self.dvrk_kin = dvrkKinematics()

    def set_pose(self, pos1=[], rot1=[], pos2=[], rot2=[], wait_callback=True):
        assert not np.isnan(np.sum(pos1))
        assert not np.isnan(np.sum(pos2))
        assert not np.isnan(np.sum(rot1))
        assert not np.isnan(np.sum(rot2))
        msg1 = Pose()
        msg2 = Pose()
        if pos1==[]:
            pos_cmd1, _ = self.arm1.get_current_pose(wait_callback=True)
        else:
            pos_cmd1 = pos1
        msg1.position.x = pos_cmd1[0]
        msg1.position.y = pos_cmd1[1]
        msg1.position.z = pos_cmd1[2]

        if rot1==[]:
            _, rot_cmd1 = self.arm1.get_current_pose(wait_callback=True)
        else:
            rot_cmd1 = rot1
        rot_transformed1 = self.arm1.set_rot_transform(rot_cmd1)
        msg1.orientation.x = rot_transformed1[0]
        msg1.orientation.y = rot_transformed1[1]
        msg1.orientation.z = rot_transformed1[2]
        msg1.orientation.w = rot_transformed1[3]

        if pos2 == []:
            pos_cmd2, _ = self.arm2.get_current_pose(wait_callback=True)
        else:
            pos_cmd2 = pos2
        msg2.position.x = pos_cmd2[0]
        msg2.position.y = pos_cmd2[1]
        msg2.position.z = pos_cmd2[2]

        if rot2 == []:
            _, rot_cmd2 = self.arm2.get_current_pose(wait_callback=True)
        else:
            rot_cmd2 = rot2
        rot_transformed2 = self.arm2.set_rot_transform(rot_cmd2)
        msg2.orientation.x = rot_transformed2[0]
        msg2.orientation.y = rot_transformed2[1]
        msg2.orientation.z = rot_transformed2[2]
        msg2.orientation.w = rot_transformed2[3]

        if wait_callback:
            self.arm1._dvrkArm__goal_reached_event.clear()
            self.arm2._dvrkArm__goal_reached_event.clear()
            self.arm1._dvrkArm__set_position_goal_cartesian_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_cartesian_pub.publish(msg2)
            return self.arm1._dvrkArm__goal_reached_event.wait(10) and self.arm2._dvrkArm__goal_reached_event.wait(10)  # 10 seconds at most:
        else:
            self.arm1._dvrkArm__set_position_goal_cartesian_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_cartesian_pub.publish(msg2)
            return True

    def set_joint(self, joint1=[], joint2=[], wait_callback=True):
        assert not np.isnan(np.sum(joint1))
        assert not np.isnan(np.sum(joint2))
        msg1 = JointState()
        msg2 = JointState()
        if joint1==[]:
            msg1.position = self.arm1.get_current_joint(wait_callback=True)
        else:
            msg1.position = joint1

        if joint2==[]:
            msg2.position = self.arm2.get_current_joint(wait_callback=True)
        else:
            msg2.position = joint2
        if wait_callback:
            self.arm1._dvrkArm__goal_reached_event.clear()
            self.arm2._dvrkArm__goal_reached_event.clear()
            self.arm1._dvrkArm__set_position_goal_joint_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_joint_pub.publish(msg2)
            return self.arm1._dvrkArm__goal_reached_event.wait(10) and self.arm2._dvrkArm__goal_reached_event.wait(10)    # 10 seconds at most
        else:
            self.arm1._dvrkArm__set_position_goal_joint_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_joint_pub.publish(msg2)
            return True

    def set_jaw(self, jaw1=[], jaw2=[], wait_callback=True):
        assert not np.isnan(np.sum(jaw1))
        assert not np.isnan(np.sum(jaw2))
        msg1 = JointState()
        msg2 = JointState()
        if jaw1==[]:
            msg1.position = self.arm1.get_current_jaw(wait_callback=True)
        else:
            msg1.position = jaw1

        if jaw2==[]:
            msg2.position = self.arm2.get_current_jaw(wait_callback=True)
        else:
            msg2.position = jaw2

        if wait_callback:
            self.arm1._dvrkArm__goal_reached_event.clear()
            self.arm2._dvrkArm__goal_reached_event.clear()
            self.arm1._dvrkArm__set_position_goal_jaw_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_jaw_pub.publish(msg2)
            return self.arm1._dvrkArm__goal_reached_event.wait(10) and self.arm2._dvrkArm__goal_reached_event.wait(10)  # 10 seconds at most
        else:
            self.arm1._dvrkArm__set_position_goal_jaw_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_jaw_pub.publish(msg2)
            return True

    def set_arm_position(self, pos1=[], pos2=[], wait_callback=True):
        assert not np.isnan(np.sum(pos1))
        assert not np.isnan(np.sum(pos2))
        msg1 = JointState()
        msg2 = JointState()
        if pos1==[]:
            msg1.position = self.arm1.get_current_joint(wait_callback=True)
        else:
            j1, j2, j3 = self.dvrk_kin.ik_position_straight(pos1)
            msg1.position = [j1, j2, j3, 0.0, 0.0, 0.0]

        if pos2==[]:
            msg2.position = self.arm2.get_current_jaw(wait_callback=True)
        else:
            j1, j2, j3 = self.dvrk_kin.ik_position_straight(pos2)
            msg2.position = [j1, j2, j3, 0.0, 0.0, 0.0]

        if wait_callback:
            self.arm1._dvrkArm__goal_reached_event.clear()
            self.arm2._dvrkArm__goal_reached_event.clear()
            self.arm1._dvrkArm__set_position_goal_joint_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_joint_pub.publish(msg2)
            return self.arm1._dvrkArm__goal_reached_event.wait(10) and self.arm2._dvrkArm__goal_reached_event.wait(10)    # 10 seconds at most
        else:
            self.arm1._dvrkArm__set_position_goal_joint_pub.publish(msg1)
            self.arm2._dvrkArm__set_position_goal_joint_pub.publish(msg2)
            return True

    def get_pose(self):
        return (self.arm1._dvrkArm__act_pos, self.arm1.get_rot_transform(self.arm1._dvrkArm__act_rot),
                self.arm1._dvrkArm__act_jaw), \
               (self.arm2._dvrkArm__act_pos, self.arm2.get_rot_transform(self.arm2._dvrkArm__act_rot),
                self.arm2._dvrkArm__act_jaw)

    def get_joint(self):
        return self.arm1._dvrkArm__act_joint, self.arm2._dvrkArm__act_joint

    def shutdown(self):
        self.arm1.shutdown()
        self.arm2.shutdown()


if __name__ == "__main__":
    p = dvrkDualArm()
    # p.set_jaw(jaw1=[np.deg2rad(0)], jaw2=[np.deg2rad(0)])
    pos1 = [0.14256737, 0.00919187, -0.15461194]
    pos2 = [-0.07924617, 0.01235677, -0.14679096]
    while True:
        p.set_pose(pos1=pos1, pos2=pos2)

        # p.set_joint(joint1=[0.8, -0.6, 0.12, 0.4, -0.3, -0.6])
        # p.set_joint(joint1=[-0.3, -0.6, 0.12, 0.4, -0.3, -0.6])



    # while True:
    #     pos11 = [-0.05, 0.05, -0.13]
    #     rot11 = np.array([0, 0, 0]) * np.pi / 180.  # ZYX Euler angle in (deg)
    #     q11 = U.euler_to_quaternion(rot11)
    #     jaw11 = [0*np.pi/180.]
    #     pos12 = [-0.05, 0.05, -0.13]
    #     rot12 = np.array([0, 0, 0]) * np.pi / 180.  # ZYX Euler angle in (deg)
    #     q12 = U.euler_to_quaternion(rot12)
    #     jaw12 = [0*np.pi/180.]
    #     joint11 = [0.4, -0.3, 0.18, 0.0, -1.0, 0.0]
    #     joint12 = [0.4, -0.3, 0.18, 0.0, -1.0, 0.0]
    #     # p.set_pose_interpolate(pos1=pos11, rot1=q11, pos2=pos12, rot2=q12, method='cubic')
    #     p.set_joint_interpolate(joint1=joint11, joint2=[], method='cubic')
    #     # p.set_joint(joint1=joint11)
    #
    #     # p.set_joint(joint1=[0.0, -0.4, 0.14, 0.0, 0.0, 0.0])
    #     # p.set_joint_interpolate(joint1=[0.0, -0.4, 0.14, 0.0, 0.0, 0.0], joint2=[], method='cubic')
    #
    #
    #     # p.set_jaw(jaw1=jaw11, jaw2=jaw12)
    #
    #     pos21 = [0.05, 0.05, -0.13]
    #     rot21 = np.array([0, 0, 0]) * np.pi / 180.  # ZYX Euler angle in (deg)
    #     q21 = U.euler_to_quaternion(rot21)
    #     jaw21 = [0*np.pi/180]
    #     pos22 = [0.05, 0.05, -0.13]
    #     rot22 = np.array([0, 0, 0]) * np.pi / 180.  # ZYX Euler angle in (deg)
    #     q22 = U.euler_to_quaternion(rot22)
    #     jaw22 = [0*np.pi/180]
    #     joint21 = [-0.4, -0.3, 0.18, 0.0, -1.0, 0.0]
    #     joint22 = [-0.4, -0.3, 0.18, 0.0, -1.0, 0.0]
    #     # p.set_pose_interpolate(pos1=pos21, rot1=q21, pos2=pos22, rot2=q22, method='cubic')
    #     p.set_joint_interpolate(joint1=joint21, joint2=[], method='cubic')
    #     # p.set_joint(joint1=joint21)
    #
    #     # p.set_joint(joint1=[0.0, -0.4, 0.14, 0.0, 0.0, 0.0])
    #     # p.set_joint_interpolate(joint1=[0.0, -0.4, 0.14, 0.0, 0.0, 0.0], joint2=[], method='cubic')
    #     # p.set_jaw(jaw1=jaw21, jaw2=jaw22)