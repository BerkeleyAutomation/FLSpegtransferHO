import sympy as sym
from sympy import sin, cos, simplify, pprint, diff, expand
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage, JointState
from std_msgs.msg import String


# Gravity Compensation of Master Controller
class dvrkGravityCompensation:
    def __init__(self):
        # create ROS node
        if not rospy.get_node_uri():
            rospy.init_node('GC_node', anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        # ROS
        self.__wrench1_pub = rospy.Publisher('/dvrk/MTMR/set_effort_joint', JointState, latch=True, queue_size=1)
        self.__wrench2_pub = rospy.Publisher('/dvrk/MTML/set_effort_joint', JointState, latch=True, queue_size=1)
        rospy.Subscriber('/dvrk/MTMR/state_joint_current', JointState, self.__get_MTMR_joint_cb)
        rospy.Subscriber('/dvrk/MTML/state_joint_current', JointState, self.__get_MTML_joint_cb)
        rospy.Subscriber('/dvrk/MTMR_PSM1/current_state', String, self.__get_MTMR_status_cb)
        rospy.Subscriber('/dvrk/MTML_PSM2/current_state', String, self.__get_MTML_status_cb)

        # Data members
        self.__joint1 = [0.0]*7
        self.__joint2 = [0.0]*7

        # Define constants
        self.g = 9.81
        self.m1 = 0.6
        self.m2 = 0.0
        self.m3 = 0.2
        self.L1 = 0.2794 # (m)
        self.L2 = 0.3048 # (m)
        rospy.spin()

    def __get_MTMR_joint_cb(self, data):
        self.__joint1 = list(data.position)

    def __get_MTML_joint_cb(self, data):
        self.__joint2 = list(data.position)

    def __get_MTMR_status_cb(self, data):
        if data.data == 'ALIGNING_MTM':
            print ("MTMR aligning")
        else:
            print("MTMR", data.data)
            self.__set_wrench()

    def __get_MTML_status_cb(self, data):
        if data.data == 'ALIGNING_MTM':
            print ("MTML aligning")
        else:
            print("MTML", data.data)
            self.__set_wrench()

    def __set_wrench(self):
        g = 9.81
        qR2 = self.__joint1[1]
        qR3 = self.__joint1[2]
        TR2 = self.L1 * g * self.m1 * sin(qR2) / 2 + g * self.m2 * (
                    self.L1 * sin(qR2) - self.L2 * (sin(qR2) * sin(qR3) - cos(qR2) * cos(qR3)) / 2) + g * self.m3 * (
                          self.L1 * sin(qR2) - self.L2 * (sin(qR2) * sin(qR3) - cos(qR2) * cos(qR3)))
        TR3 = -self.L2 * g * self.m2 * (sin(qR2) * sin(qR3) - cos(qR2) * cos(qR3)) / 2 - self.L2 * g * self.m3 * (
                    sin(qR2) * sin(qR3) - cos(qR2) * cos(qR3))

        qL2 = self.__joint2[1]
        qL3 = self.__joint2[2]
        TL2 = self.L1 * g * self.m1 * sin(qL2) / 2 + g * self.m2 * (
                    self.L1 * sin(qL2) - self.L2 * (sin(qL2) * sin(qL3) - cos(qL2) * cos(qL3)) / 2) + g * self.m3 * (
                          self.L1 * sin(qL2) - self.L2 * (sin(qL2) * sin(qL3) - cos(qL2) * cos(qL3)))
        TL3 = -self.L2 * g * self.m2 * (sin(qL2) * sin(qL3) - cos(qL2) * cos(qL3)) / 2 - self.L2 * g * self.m3 * (
                    sin(qL2) * sin(qL3) - cos(qL2) * cos(qL3))

        msg1 = JointState()
        msg2 = JointState()
        # these coefficients are the choices from my try-and-error
        msg1.effort = np.array([0.0, TR2*1.3, TR3, 0.0, 0.0, 0.0, 0.0])*1.0
        msg2.effort = np.array([0.0, TL2*1.1, TL3, 0.0, 0.0, 0.0, 0.0])*0.6
        print(msg1.effort)
        print(msg2.effort)
        self.__wrench1_pub.publish(msg1)
        self.__wrench2_pub.publish(msg2)

    @classmethod
    def DH_transform(cls, a, alpha, d, theta):
        T = sym.Matrix([[cos(theta), -sin(theta), 0, a],
                        [sin(theta) * cos(alpha), cos(theta) * cos(alpha), -sin(alpha), -sin(alpha) * d],
                        [sin(theta) * sin(alpha), cos(theta) * sin(alpha), cos(alpha), cos(alpha) * d],
                        [0, 0, 0, 1]])
        return T

    def calculate_gravity_torque(self):
        """
        Symbolic derivation of gravity torque
        """
        L1, L2 = sym.symbols('L1, L2')
        q2, q3 = sym.symbols('q2, q3')
        m1, m2, m3 = sym.symbols('m1, m2, m3')
        g = sym.symbols('g')
        T01 = self.DH_transform(0, 0, 0, -q2 - sym.pi / 2)
        T1m1 = self.DH_transform(L1/2, 0, 0, 0)
        T12 = self.DH_transform(L1, 0, 0, -q3 + sym.pi / 2)
        T2m2 = self.DH_transform(-L2/2, 0, 0, 0)
        T2m3 = self.DH_transform(-L2, 0, 0, 0)
        p_m1y = (T01 * T1m1)[1, 3]
        p_m2y = (T01 * T12 * T2m2)[1, 3]
        p_m3y = (T01 * T12 * T2m3)[1, 3]
        # print(p_m1y)
        # print(p_m2y)
        # print(p_m3y)

        U1 = m1*g*(p_m1y - p_m1y.subs([(q2, 0), (q3, 0)]))
        U2 = m2*g*(p_m2y - p_m2y.subs([(q2, 0), (q3, 0)]))
        U3 = m3*g*(p_m3y - p_m3y.subs([(q2, 0), (q3, 0)]))
        U = U1 + U2 + U3    # total potential energy
        T2 = diff(U, q2)    # gravity torque
        T3 = diff(U, q3)

        print("T2=", T2)
        print("T3=", T3)
        # print(T2.subs([(q2, 0), (q3, 0)]))    # for test
        # print(T3.subs([(q2, 0), (q3, 0)]))


if __name__ == "__main__":
    gc = dvrkGravityCompensation()
    # gc.calculate_gravity_torque()