import numpy as np
import threading
import PyKDL
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf_conversions import posemath
import FLSpegtransfer.utils.CmnUtil as U
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics


class dvrkArm(object):
    """Simple arm API wrapping around ROS messages
    """
    def __init__(self, arm_name, ros_namespace='/dvrk'):
        self.dvrk_model = dvrkKinematics()

        # continuous publish from dvrk_bridge
        # actual(current) values
        self.__act_pose_frame = PyKDL.Frame()
        self.__act_pos = []
        self.__act_rot = []     # quaternion
        self.__act_jaw = []
        self.__act_joint = []
        self.__act_motor_current = [0.0]*7

        # data members, event based
        self.__arm_name = arm_name
        self.__ros_namespace = ros_namespace
        self.__goal_reached_event = threading.Event()
        self.__get_position_event = threading.Event()
        self.__get_joint_event = threading.Event()
        self.__get_jaw_event = threading.Event()
        self.__get_motor_current_event = threading.Event()

        self.__sub_list = []
        self.__pub_list = []

        # publisher
        self.__full_ros_namespace = self.__ros_namespace + self.__arm_name
        self.__set_position_joint_pub = rospy.Publisher(self.__full_ros_namespace + '/set_position_joint', JointState,
                                                        latch = True, queue_size = 1)
        self.__set_position_goal_joint_pub = rospy.Publisher(self.__full_ros_namespace + '/set_position_goal_joint',
                                                             JointState, latch = True, queue_size = 1)
        self.__set_position_cartesian_pub = rospy.Publisher(self.__full_ros_namespace
                                                            + '/set_position_cartesian',
                                                            Pose, latch = True, queue_size = 1)
        self.__set_position_goal_cartesian_pub = rospy.Publisher(self.__full_ros_namespace
                                                                 + '/set_position_goal_cartesian',
                                                                 Pose, latch = True, queue_size = 1)
        self.__set_position_jaw_pub = rospy.Publisher(self.__full_ros_namespace
                                                      + '/set_position_jaw',
                                                      JointState, latch = True, queue_size = 1)
        self.__set_position_goal_jaw_pub = rospy.Publisher(self.__full_ros_namespace
                                                           + '/set_position_goal_jaw',
                                                           JointState, latch = True, queue_size = 1)

        self.__pub_list = [self.__set_position_joint_pub,
                           self.__set_position_goal_joint_pub,
                           self.__set_position_cartesian_pub,
                           self.__set_position_goal_cartesian_pub,
                           self.__set_position_jaw_pub,
                           self.__set_position_goal_jaw_pub]

        self.__sub_list = [rospy.Subscriber(self.__full_ros_namespace + '/goal_reached',
                                          Bool, self.__goal_reached_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/position_cartesian_current',
                                          PoseStamped, self.__position_cartesian_current_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/state_joint_current',
                                            JointState, self.__position_joint_current_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/state_jaw_current',
                                            JointState, self.__position_jaw_current_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/io/actuator_current_measured',
                                            JointState, self.__motor_current_measured_cb)]

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('dvrkArm_node', anonymous = True, log_level = rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        # wait until these are not empty
        self.__act_pos, _ = self.get_current_pose(wait_callback=True)
        self.__act_jaw = self.get_current_jaw(wait_callback=True)
        self.__act_joint = self.get_current_joint(wait_callback=True)

    def shutdown(self):
        rospy.signal_shutdown("Shutdown signal received.")

    """
    Callback function
    """
    def __goal_reached_cb(self, data):
        self.__goal_reached = data.data
        self.__goal_reached_event.set()

    def __position_cartesian_current_cb(self, data):
        self.__act_pose_frame = posemath.fromMsg(data.pose)
        self.__act_pos = [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        self.__act_rot = [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w]
        self.__get_position_event.set()

    def __position_joint_current_cb(self, data):
        self.__act_joint = list(data.position)
        self.__get_joint_event.set()

    def __position_jaw_current_cb(self, data):
        self.__act_jaw = list(data.position)
        self.__get_jaw_event.set()

    def __motor_current_measured_cb(self, data):
        self.__act_motor_current = list(data.position)
        self.__get_motor_current_event.set()


    """
    Get function
    """
    def get_current_pose(self, wait_callback=False):
        if wait_callback:
            self.__get_position_event.clear()
            if self.__get_position_event.wait(20):  # 20 seconds at most
                return self.__act_pos, self.get_rot_transform(self.__act_rot)
            else:
                return []
        else:
            return self.__act_pos, self.get_rot_transform(self.__act_rot)

    def get_current_pose_frame(self):
        return self.__act_pose_frame

    def get_current_joint(self, wait_callback=False):
        if wait_callback:
            self.__get_joint_event.clear()
            if self.__get_joint_event.wait(20):  # 20 seconds at most
                joint = self.__act_joint
                return joint
            else:
                return []
        else:
            joint = self.__act_joint
            return joint

    def get_current_jaw(self, wait_callback=False):
        if wait_callback:
            self.__get_jaw_event.clear()
            if self.__get_jaw_event.wait(20):   # 20 seconds at most
                jaw = self.__act_jaw
                return jaw
            else:
                return []
        else:
            jaw = self.__act_jaw
            return jaw

    def get_motor_current(self, wait_callback=False):
        if wait_callback:
            self.__get_motor_current_event.clear()
            if self.__get_motor_current_event.wait(20):     # 20 seconds at most
                motor_current = self.__act_motor_current
                return motor_current
            else:
                return []
        else:
            motor_current = self.__act_motor_current
            return motor_current

    """
    Set function
    """
    def set_pose(self, pos=[], rot=[], use_ik=True, wait_callback=True):
        assert not np.isnan(np.sum(pos))
        assert not np.isnan(np.sum(rot))
        msg = Pose()
        if pos==[]:
            pos_cmd, _ = self.get_current_pose(wait_callback=True)
        else:
            pos_cmd = pos
        msg.position.x = pos_cmd[0]
        msg.position.y = pos_cmd[1]
        msg.position.z = pos_cmd[2]

        if rot==[]:
            _, rot_cmd = self.get_current_pose(wait_callback=True)
        else:
            rot_cmd = rot
        rot_transformed = self.set_rot_transform(rot_cmd)
        msg.orientation.x = rot_transformed[0]
        msg.orientation.y = rot_transformed[1]
        msg.orientation.z = rot_transformed[2]
        msg.orientation.w = rot_transformed[3]

        if use_ik:
            # convert to joint angles using IK
            joint = self.dvrk_model.pose_to_joint(pos, rot)
            return self.set_joint(joint, wait_callback=wait_callback)
        else:
            if wait_callback:
                self.__goal_reached_event.clear()
                self.__set_position_goal_cartesian_pub.publish(msg)
                self.__goal_reached_event.wait()  # 10 seconds at most:
                return True
            else:
                self.__set_position_goal_cartesian_pub.publish(msg)
                return True

    # specify intermediate points between q0 & qf
    def set_pose_interpolate(self, pos=[], rot=[], method='LSPB', t_step=0.02):
        assert not np.isnan(np.sum(pos))
        assert not np.isnan(np.sum(rot))

        # Define q0 and qf
        pos0, rot0 = self.get_current_pose(wait_callback=True)
        if len(pos)==0:
            posf = pos0
        else:
            posf = pos
        if len(rot)==0:
            rotf = rot0
        else:
            rotf = rot
        rot0 = U.quaternion_to_euler(rot0)
        rotf = U.quaternion_to_euler(rotf)
        q0 = np.concatenate((pos0, rot0))
        qf = np.concatenate((posf, rotf))

        # Define trajectory
        if np.allclose(q0, qf):
            return False
        else:
            # v_max = [0.1, 0.1, 0.3, 0.5, 0.5, 0.5]
            # a_max = [0.1, 0.1, 0.3, 0.5, 0.5, 0.5]
            if method=='cubic':
                t, traj = self.cubic(q0, qf, v_max=dvrkVar.v_max, a_max=dvrkVar.a_max, t_step=t_step)
            elif method=='LSPB':
                t, traj = self.LSPB(q0, qf, v_max=dvrkVar.v_max, a_max=dvrkVar.a_max, t_step=t_step)
            else:
                raise IndexError

            # Execute trajectory
            for q in traj:
                self.set_pose(q[:3], U.euler_to_quaternion(q[3:]), wait_callback=False)
                rospy.sleep(t_step)
            self.set_pose(traj[-1][:3], U.euler_to_quaternion(traj[-1][3:]), wait_callback=True)
            return True

    def set_joint(self, joint, wait_callback=True):
        assert not np.isnan(np.sum(joint))
        msg = JointState()
        msg.position = joint
        if wait_callback:
            self.__goal_reached_event.clear()
            self.__set_position_goal_joint_pub.publish(msg)
            return self.__goal_reached_event.wait()  # 20 seconds at most
        else:
            self.__set_position_goal_joint_pub.publish(msg)
            return True

    def set_joint_interpolate(self, joint, method='LSPB', t_step=0.01):
        assert not np.isnan(np.sum(joint))
        # Define q0 and qf
        q0 = self.get_current_joint(wait_callback=True)
        if len(joint) == 0:
            qf = q0
        else:
            qf = joint

        if np.allclose(q0, qf):
            return False
        else:
            # Define trajectory
            # v_max = [2.0, 2.0, 0.2, 10.0, 10.0, 10.0]
            # a_max = [2.0, 2.0, 0.2, 10.0, 10.0, 10.0]
            if method=='cubic':
                t, traj = self.cubic(q0, qf, v_max=dvrkVar.v_max, a_max=dvrkVar.a_max, t_step=t_step)
            elif method=='LSPB':
                t, traj = self.LSPB(q0, qf, v_max=dvrkVar.v_max, a_max=dvrkVar.a_max, t_step=t_step)
            else:
                raise IndexError

            import time
            # Execute trajectory
            for q in traj:
                self.set_joint(q, wait_callback=False)
                rospy.sleep(t_step)
            self.set_joint(traj[-1], wait_callback=True)
            return True

    def set_jaw(self, jaw, wait_callback=True):
        assert not np.isnan(np.sum(jaw))
        msg = JointState()
        msg.position = jaw
        if wait_callback:
            self.__goal_reached_event.clear()
            self.__set_position_goal_jaw_pub.publish(msg)
            self.__goal_reached_event.wait()  # 10 seconds at most
            return True
        else:
            self.__set_position_goal_jaw_pub.publish(msg)
            return True

    # this function doesn't issue the "goal_reached" flag at the end
    # Not reliable to use with set_pose or set_joint
    def set_jaw_interpolate(self, jaw, t_step=0.01):
        assert not np.isnan(np.sum(jaw))

        # Define q0 and qf
        jaw0 = self.get_current_jaw(wait_callback=True)
        if len(jaw) == 0:
            jawf = jaw0
        else:
            jawf = jaw
        q0 = jaw0
        qf = jawf

        # Define trajectory
        if np.allclose(q0, qf):
            return False
        else:
            t, traj = self.Linear(q0, qf, v=[1.0], t_step=t_step)
            # Execute trajectory
            for q in traj:
                self.set_jaw(q, wait_callback=False)
                rospy.sleep(t_step)
            # self.set_jaw(qf, wait_callback=True)
            return True

    """
    Conversion function
    """
    # Matching coordinate of the robot base and the end effector
    def set_rot_transform(self, q):
        qx, qy, qz, qw = q
        R1 = PyKDL.Rotation.Quaternion(qx,qy,qz,qw)
        R2 = PyKDL.Rotation.EulerZYX(-np.pi/2, 0, 0)  # rotate -90 (deg) around z-axis
        R3 = PyKDL.Rotation.EulerZYX(0, np.pi, 0)  # rotate 180 (deg) around y-axis
        R = R1 * R2 * R3
        return R.GetQuaternion()

    # Matching coordinate of the robot base and the end effector
    def get_rot_transform(self, q):
        qx, qy, qz, qw = q
        R1 = PyKDL.Rotation.Quaternion(qx, qy, qz, qw)
        R2 = PyKDL.Rotation.EulerZYX(0, np.pi, 0)  # rotate 180 (deg) around y-axis
        R3 = PyKDL.Rotation.EulerZYX(-np.pi/2, 0, 0)  # rotate -90 (deg) around z-axis
        R = R1 * R2.Inverse() * R3.Inverse()
        return R.GetQuaternion()


    """
    Trajectory
    """
    def Linear(self, q0, qf, v, t_step):
        num_axis = len(q0)
        q0 = np.array(q0)
        qf = np.array(qf)
        v = np.array(v)

        if np.allclose(q0, qf):
            t = [0.0]
            joint = [qf]
            return t, joint

        # Design variables
        tf = abs((qf - q0) / v)  # total time taken

        # Calculate trajectories
        t = np.arange(start=0.0, stop=tf, step=t_step)
        joint = []
        for i in range(num_axis):
            # joint traj.
            q = (qf[i]-q0[i])/tf[i] * t + q0[i]
            joint.append(q)
        joint = np.array(joint).T
        assert ~np.isnan(t).any()
        assert ~np.isnan(joint).any()
        return t, joint

    # q0, qf could be cartesian coordinates or joint configurations
    @classmethod
    def LSPB(cls, q0, qf, v_max, a_max, t_step=0.01):
        num_axis = len(q0)
        q0 = np.array(q0)
        qf = np.array(qf)
        v_max = np.array(v_max)
        a_max = np.array(a_max)
        if np.allclose(q0, qf):
            t = [0.0]
            joint = [qf]
            return t, joint

        # Design variables
        A = max(abs((qf - q0) / a_max))
        B = max(abs((qf - q0) / v_max))
        tb = A / B
        tf = B + tb
        if tf < 2 * tb:
            tb = np.sqrt(A)
            tf = 2 * tb

        # Define coefficients
        A = np.array([[tb ** 2, -tb, -1, 0.0, 0.0, 0.0],
                      [2 * tb, -1, 0.0, 0.0, 0.0, 0.0],
                      [0.0, tf - tb, 1, -(tf - tb) ** 2, -(tf - tb), -1],
                      [0.0, 1.0, 0.0, -2 * (tf - tb), -1, 0.0],
                      [0.0, 0.0, 0.0, 2 * tf, 1.0, 0.0],
                      [0.0, 0.0, 0.0, tf ** 2, tf, 1.0]])
        b = np.block([[-q0], [np.zeros_like(q0)], [np.zeros_like(q0)], [np.zeros_like(q0)], [np.zeros_like(q0)], [qf]])
        coeff = np.linalg.inv(A).dot(b)
        a1 = coeff[0]
        a2 = coeff[1]
        b2 = coeff[2]
        a3 = coeff[3]
        b3 = coeff[4]
        c3 = coeff[5]

        # Calculate trajectories
        t = np.arange(start=0.0, stop=tf, step=t_step)
        t1 = t[t < tb]
        t2 = t[(tb <= t) & (t < tf - tb)]
        t3 = t[tf - tb <= t]
        joint = []
        for i in range(num_axis):
            # joint traj.
            traj1 = a1[i] * t1 ** 2 + q0[i]
            traj2 = a2[i] * t2 + b2[i]
            traj3 = a3[i] * t3 ** 2 + b3[i] * t3 + c3[i]
            q = np.concatenate((traj1, traj2, traj3))
            joint.append(q)
        joint = np.array(joint).T
        assert ~np.isnan(t).any()
        assert ~np.isnan(joint).any()
        return t, joint

    @classmethod
    def cubic(cls, q0, qf, v_max, a_max, t_step=0.01):
        num_axis = len(q0)
        q0 = np.array(q0)
        qf = np.array(qf)
        v_max = np.array(v_max)
        a_max = np.array(a_max)
        if np.allclose(q0, qf):
            t = [0.0]
            joint = [qf]
            return t, joint

        # v_max = 1.5*(qf-q0)/tf
        tf_vel = 1.5 * (qf - q0) / v_max

        # a_max = 6*(qf-q0)/(tf**2)
        tf_acc = np.sqrt(abs(6 * (qf - q0) / a_max))
        tf_Rn = np.maximum(tf_vel, tf_acc)  # tf for each axis (nx1 array)
        tf = max(tf_Rn)  # maximum scalar value among axes

        # Define coefficients
        a = -2 * (qf - q0) / (tf ** 3)
        b = 3 * (qf - q0) / (tf ** 2)
        c = np.zeros_like(a)
        d = q0

        # Calculate trajectorie
        t = np.arange(start=0.0, stop=tf, step=t_step)
        joint = []
        for i in range(num_axis):
            # joint traj.
            q = a[i] * t ** 3 + b[i] * t ** 2 + c[i] * t + d[i]
            joint.append(q)
        joint = np.array(joint).T
        assert ~np.isnan(t).any()
        assert ~np.isnan(joint).any()
        return t, joint

    @classmethod
    def cubic_time(cls, q0, qf, tf, t_step=0.01):
        num_axis = len(q0)
        q0 = np.array(q0)
        qf = np.array(qf)
        if np.allclose(q0, qf):
            t = [0.0]
            joint = [qf]
            return t, joint

        # Define coefficients
        a = -2 * (qf - q0) / (tf ** 3)
        b = 3 * (qf - q0) / (tf ** 2)
        c = np.zeros_like(a)
        d = q0

        # Calculate trajectorie
        t = np.arange(start=0.0, stop=tf, step=t_step)
        joint = []
        for i in range(num_axis):
            # joint traj.
            q = a[i] * t ** 3 + b[i] * t ** 2 + c[i] * t + d[i]
            joint.append(q)
        joint = np.array(joint).T
        assert ~np.isnan(t).any()
        assert ~np.isnan(joint).any()
        return t, joint


if __name__ == "__main__":
    arm1 = dvrkArm('/PSM1')
    arm2 = dvrkArm('/PSM2')
    # pos1 = [0.12, 0.0, -0.13]
    # rot1 = [0.0, 0.0, 0.0]
    # q1 = U.euler_to_quaternion(rot1, unit='deg')
    # jaw1 = [30 * np.pi / 180.]
    # pos2 = [-0.12, 0.0, -0.13]
    # rot2 = [0.0, 0.0, 0.0]
    # q2 = U.euler_to_quaternion(rot2, unit='deg')
    # jaw2 = [0.0]
    joint1 = [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]
    joint2 = [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]
    # p1.set_joint(joint=joint1)
    # p1.set_joint(joint=joint2)
    while True:
        # p1.set_joint(joint=joint1)
        # p1.set_joint(joint=joint2)
        # p1.set_pose_interpolate(pos=pos1, rot=q1, method='LSPB')
        # p1.set_pose_interpolate(pos=pos2, rot=q2, method='LSPB')
        # print ("moved")
        arm1.set_joint_interpolate(joint=joint1, method='LSPB')
        arm1.set_joint_interpolate(joint=joint2, method='LSPB')
        arm1.set_joint(joint=joint1)
        arm1.set_joint(joint=joint2)
        # print ("moved")