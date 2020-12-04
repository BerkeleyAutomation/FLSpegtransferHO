# from robopy.base.serial_link import SerialLink
# from robopy.base.serial_link import Revolute
# from robopy.base.serial_link import Prismatic
from math import pi
import numpy as np
# from robopy import transforms as tr
# from robopy import graphics
import FLSpegtransfer.utils.CmnUtil as U
import FLSpegtransfer.motion.dvrkVariables as dvrkVar


# class dvrkKinematics(SerialLink):
#     def __init__(self):
#         # self.qn = np.matrix([[0, pi / 4, pi, 0, pi / 4, 0]])
#         # self.qr = np.matrix([[0, pi / 2, -pi / 2, 0, 0, 0]])
#         # self.qz = np.matrix([[0, 0, 0, 0, 0, 0]])
#         # self.qs = np.matrix([[0, 0, -pi / 2, 0, 0, 0]])
#         # self.scale = 1
#         # param = {
#         #     "cube_axes_x_bounds": np.matrix([[-1.5, 1.5]]),
#         #     "cube_axes_y_bounds": np.matrix([[-0.7, 1.5]]),
#         #     "cube_axes_z_bounds": np.matrix([[-1.5, 1.5]]),
#         #     "floor_position": np.matrix([[0, -0.7, 0]])
#         # }
#         links = [
#             Revolute(j=0, theta=0, d=0, a=0, alpha=-pi/2, offset=pi/2, qlim=(-90 * np.pi / 180, 90 * np.pi / 180)),
#             Revolute(j=0, theta=0, d=0, a=0, alpha=-pi/2, offset=pi/2, qlim=(-60 * np.pi / 180, 60 * np.pi / 180)),
#             Prismatic(j=0, theta=0, d=0, a=0, alpha=0, offset=-dvrkVar.L1+dvrkVar.L2, qlim=(0, 0.5)),
#             Revolute(j=0, theta=0, d=0, a=0, alpha=pi/2, offset=0, qlim=(-90 * np.pi / 180, 90 * np.pi / 180)),
#             Revolute(j=0, theta=0, d=0, a=dvrkVar.L3, alpha=-pi/2, offset=pi/2, qlim=(-90 * np.pi / 180, 90 * np.pi / 180)),
#             Revolute(j=0, theta=0, d=0, a=dvrkVar.L4, alpha=0, offset=0, qlim=(-90 * np.pi / 180, 90 * np.pi / 180))]
#
#         base = tr.trotx(90, unit='deg')
#         tool = tr.trotz(90, unit='deg')*tr.trotx(-90, unit='deg')
#         # tool = tr.trotz(-90, unit='deg') * tr.trotx(-90, unit='deg')
#
#         # def __init__(self, links, name=None, base=None, tool=None, stl_files=None, q=None, colors=None, param=None):
#
#         file_names = SerialLink._setup_file_names(7)
#         colors = graphics.vtk_named_colors(["Red", "DarkGreen", "Blue", "Cyan", "Magenta", "Yellow", "White"])
#
#         super().__init__(links=links, name='dvrk', base=base, tool=tool, colors=colors)

class dvrkKinematics():
    @classmethod
    def pose_to_transform(cls, pos, quat):
        """

        :param pos: position (m)
        :param rot: quaternion (qx, qy, qz, qw)
        :return:
        """
        T = np.zeros((4, 4))
        R = U.quaternion_to_R(quat)
        T[:3,:3] = R
        T[:3,-1] = np.transpose(pos)
        T[-1,-1] = 1
        return T

    def pose_to_joint(self, pos, rot, method='analytic'):
        if pos==[] or rot==[]:
            joint = []
        else:
            T = self.pose_to_transform(pos, rot)    # current transformation
            joint = self.ik(T, method)
        return joint

    @classmethod
    def DH_transform(cls, a, alpha, d, theta, unit='rad'):  # modified DH convention
        if unit == 'deg':
            alpha = np.deg2rad(alpha)
            theta = np.deg2rad(theta)
        T = np.array([[np.cos(theta), -np.sin(theta), 0, a],
                      [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha),
                       -np.sin(alpha) * d],
                      [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                      [0, 0, 0, 1]])
        return T

    @classmethod
    def fk(cls, joints, L1=0, L2=0, L3=0, L4=0):
        q1, q2, q3, q4, q5, q6 = joints
        T01 = dvrkKinematics.DH_transform(0, np.pi / 2, 0, q1 + np.pi / 2)
        T12 = dvrkKinematics.DH_transform(0, -np.pi / 2, 0, q2 - np.pi / 2)
        T23 = dvrkKinematics.DH_transform(0, np.pi / 2, q3 - L1 + L2, 0)
        T34 = dvrkKinematics.DH_transform(0, 0, 0, q4)
        T45 = dvrkKinematics.DH_transform(0, -np.pi / 2, 0, q5 - np.pi / 2)
        T56 = dvrkKinematics.DH_transform(L3, -np.pi / 2, 0, q6 - np.pi / 2)
        T67 = dvrkKinematics.DH_transform(0, -np.pi / 2, L4, 0)
        T78 = dvrkKinematics.DH_transform(0, np.pi, 0, np.pi)
        T08 = T01.dot(T12).dot(T23).dot(T34).dot(T45).dot(T56).dot(T67).dot(T78)
        return T08

    def ik(self, T, method='analytic'):
        if method=='analytic':
            T = np.linalg.inv(T)
            x84 = T[0, 3]
            y84 = T[1, 3]
            z84 = T[2, 3]
            q6 = np.arctan2(x84, z84 - dvrkVar.L4)
            temp = -dvrkVar.L3 + np.sqrt(x84 ** 2 + (z84 - dvrkVar.L4) ** 2)
            q3 = dvrkVar.L1 - dvrkVar.L2 + np.sqrt(y84 ** 2 + temp ** 2)
            q5 = np.arctan2(-y84, temp)
            R84 = np.array([[np.sin(q5) * np.sin(q6), -np.cos(q6), np.cos(q5) * np.sin(q6)],
                            [np.cos(q5), 0, -np.sin(q5)],
                            [np.cos(q6) * np.sin(q5), np.sin(q6), np.cos(q5) * np.cos(q6)]])
            R80 = T[:3, :3]
            R40 = R84.T.dot(R80)
            n32 = R40[2, 1]
            n31 = R40[2, 0]
            n33 = R40[2, 2]
            n22 = R40[1, 1]
            n12 = R40[0, 1]
            q2 = np.arcsin(n32)
            q1 = np.arctan2(-n31, n33)
            q4 = np.arctan2(n22, n12)
            joint = [q1, q2, q3, q4, q5, q6]  # q3 and q4 are swapped
        elif method=='numerical':
            q0 = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # initial guess
            ik_sol = self.ikine(T, q0)
            joint = [ik_sol[0, 0], ik_sol[0, 1], ik_sol[0, 2], ik_sol[0, 3], ik_sol[0, 4], ik_sol[0, 5]]
        assert ~np.isnan(joint).any()
        return joint

    @classmethod
    def ik_position_straight(cls, pos, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=dvrkVar.L4):
        # L1: Rcc (m)
        # L2: length of tool (m)
        x = pos[0]      # (m)
        y = pos[1]
        z = pos[2]

        # Forward Kinematics
        # x = np.cos(q2)*np.sin(q0)*(L2-L1+q3)
        # y = -np.sin(q2)*(L2-L1+q3)
        # z = -np.cos(q0)*np.cos(q2)*(L2-L1+q3)

        # Inverse Kinematics
        q1 = np.arctan2(x, -z)     # (rad)
        q2 = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))  # (rad)
        q3 = np.sqrt(x ** 2 + y ** 2 + z ** 2) + L1 - (L2+L3+L4)   # (m)
        return q1, q2, q3

    @classmethod
    def ik_orientation(cls, q1,q2,Rb):
        R03 = np.array([[-np.sin(q1)*np.sin(q2), -np.cos(q1), np.cos(q2)*np.sin(q1)],
                        [-np.cos(q2), 0, -np.sin(q2)],
                        [np.cos(q1)*np.sin(q2), -np.sin(q1), -np.cos(q1)*np.cos(q2)]])
        R38 = R03.T.dot(Rb)
        r12 = R38[0,1]
        r22 = R38[1,1]
        r31 = R38[2,0]
        r32 = R38[2,1]
        r33 = R38[2,2]
        q4 = np.arctan2(-r22, -r12)     # (rad)
        q6 = np.arctan2(-r31, -r33)
        q5 = np.arctan2(r32, np.sqrt(r31**2+r33**2))
        return q4,q5,q6

    @classmethod
    def jacobian(cls, joints, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=dvrkVar.L4):
        q1,q2,q3,q4,q5,q6 = joints
        J11 = L2*np.cos(q1)*np.cos(q2) - L1*np.cos(q1)*np.cos(q2) + q3*np.cos(q1)*np.cos(q2) + L3*np.cos(q1)*np.cos(q2)*np.cos(q5) - L4*np.cos(q4)*np.sin(q1)*np.sin(q6) + L3*np.sin(q1)*np.sin(q4)*np.sin(q5) - L3*np.cos(q1)*np.cos(q4)*np.sin(q2)*np.sin(q5) - L4*np.cos(q1)*np.sin(q2)*np.sin(q4)*np.sin(q6) + L4*np.cos(q6)*np.sin(q1)*np.sin(q4)*np.sin(q5) + L4*np.cos(q1)*np.cos(q2)*np.cos(q5)*np.cos(q6) - L4*np.cos(q1)*np.cos(q4)*np.cos(q6)*np.sin(q2)*np.sin(q5)
        J12 = -np.sin(q1)*(L2*np.sin(q2) - L1*np.sin(q2) + q3*np.sin(q2) + L3*np.cos(q5)*np.sin(q2) + L3*np.cos(q2)*np.cos(q4)*np.sin(q5) + L4*np.cos(q5)*np.cos(q6)*np.sin(q2) + L4*np.cos(q2)*np.sin(q4)*np.sin(q6) + L4*np.cos(q2)*np.cos(q4)*np.cos(q6)*np.sin(q5))
        J13 = np.cos(q2)*np.sin(q1)
        J14 = L3*np.sin(q1)*np.sin(q2)*np.sin(q4)*np.sin(q5) - L4*np.cos(q1)*np.sin(q4)*np.sin(q6) - L4*np.cos(q1)*np.cos(q4)*np.cos(q6)*np.sin(q5) - L4*np.cos(q4)*np.sin(q1)*np.sin(q2)*np.sin(q6) - L3*np.cos(q1)*np.cos(q4)*np.sin(q5) + L4*np.cos(q6)*np.sin(q1)*np.sin(q2)*np.sin(q4)*np.sin(q5)
        J15 = -(L3 + L4*np.cos(q6))*(np.cos(q1)*np.cos(q5)*np.sin(q4) + np.cos(q2)*np.sin(q1)*np.sin(q5) + np.cos(q4)*np.cos(q5)*np.sin(q1)*np.sin(q2))
        J16 = L4*np.cos(q1)*np.cos(q4)*np.cos(q6) - L4*np.cos(q2)*np.cos(q5)*np.sin(q1)*np.sin(q6) - L4*np.cos(q6)*np.sin(q1)*np.sin(q2)*np.sin(q4) + L4*np.cos(q1)*np.sin(q4)*np.sin(q5)*np.sin(q6) + L4*np.cos(q4)*np.sin(q1)*np.sin(q2)*np.sin(q5)*np.sin(q6)
        J21 = 0
        J22 = L1*np.cos(q2) - L2*np.cos(q2) - q3*np.cos(q2) - L3*np.cos(q2)*np.cos(q5) - L4*np.cos(q2)*np.cos(q5)*np.cos(q6) + L3*np.cos(q4)*np.sin(q2)*np.sin(q5) + L4*np.sin(q2)*np.sin(q4)*np.sin(q6) + L4*np.cos(q4)*np.cos(q6)*np.sin(q2)*np.sin(q5)
        J23 = -np.sin(q2)
        J24 = np.cos(q2)*(L3*np.sin(q4)*np.sin(q5) - L4*np.cos(q4)*np.sin(q6) + L4*np.cos(q6)*np.sin(q4)*np.sin(q5))
        J25 = (L3 + L4*np.cos(q6))*(np.sin(q2)*np.sin(q5) - np.cos(q2)*np.cos(q4)*np.cos(q5))
        J26 = L4*np.cos(q5)*np.sin(q2)*np.sin(q6) - L4*np.cos(q2)*np.cos(q6)*np.sin(q4) + L4*np.cos(q2)*np.cos(q4)*np.sin(q5)*np.sin(q6)
        J31 = L2*np.cos(q2)*np.sin(q1) - L1*np.cos(q2)*np.sin(q1) + q3*np.cos(q2)*np.sin(q1) + L3*np.cos(q2)*np.cos(q5)*np.sin(q1) + L4*np.cos(q1)*np.cos(q4)*np.sin(q6) - L3*np.cos(q1)*np.sin(q4)*np.sin(q5) + L4*np.cos(q2)*np.cos(q5)*np.cos(q6)*np.sin(q1) - L4*np.cos(q1)*np.cos(q6)*np.sin(q4)*np.sin(q5) - L3*np.cos(q4)*np.sin(q1)*np.sin(q2)*np.sin(q5) - L4*np.sin(q1)*np.sin(q2)*np.sin(q4)*np.sin(q6) - L4*np.cos(q4)*np.cos(q6)*np.sin(q1)*np.sin(q2)*np.sin(q5)
        J32 = np.cos(q1)*(L2*np.sin(q2) - L1*np.sin(q2) + q3*np.sin(q2) + L3*np.cos(q5)*np.sin(q2) + L3*np.cos(q2)*np.cos(q4)*np.sin(q5) + L4*np.cos(q5)*np.cos(q6)*np.sin(q2) + L4*np.cos(q2)*np.sin(q4)*np.sin(q6) + L4*np.cos(q2)*np.cos(q4)*np.cos(q6)*np.sin(q5))
        J33 = -np.cos(q1)*np.cos(q2)
        J34 = L4*np.cos(q1)*np.cos(q4)*np.sin(q2)*np.sin(q6) - L4*np.sin(q1)*np.sin(q4)*np.sin(q6) - L3*np.cos(q4)*np.sin(q1)*np.sin(q5) - L4*np.cos(q4)*np.cos(q6)*np.sin(q1)*np.sin(q5) - L3*np.cos(q1)*np.sin(q2)*np.sin(q4)*np.sin(q5) - L4*np.cos(q1)*np.cos(q6)*np.sin(q2)*np.sin(q4)*np.sin(q5)
        J35 = (L3 + L4*np.cos(q6))*(np.cos(q1)*np.cos(q2)*np.sin(q5) - np.cos(q5)*np.sin(q1)*np.sin(q4) + np.cos(q1)*np.cos(q4)*np.cos(q5)*np.sin(q2))
        J36 = L4*np.cos(q4)*np.cos(q6)*np.sin(q1) + L4*np.cos(q1)*np.cos(q2)*np.cos(q5)*np.sin(q6) + L4*np.cos(q1)*np.cos(q6)*np.sin(q2)*np.sin(q4) + L4*np.sin(q1)*np.sin(q4)*np.sin(q5)*np.sin(q6) - L4*np.cos(q1)*np.cos(q4)*np.sin(q2)*np.sin(q5)*np.sin(q6)
        J41 = 0
        J42 = -np.cos(q1)
        J43 = 0
        J44 = np.cos(q2)*np.sin(q1)
        J45 = np.sin(q1)*np.sin(q2)*np.sin(q4) - np.cos(q1)*np.cos(q4)
        J46 = - np.cos(q1)*np.cos(q5)*np.sin(q4) - np.cos(q2)*np.sin(q1)*np.sin(q5) - np.cos(q4)*np.cos(q5)*np.sin(q1)*np.sin(q2)
        J51 = -1
        J52 = 0
        J53 = 0
        J54 = -np.sin(q2)
        J55 = np.cos(q2)*np.sin(q4)
        J56 = np.sin(q2)*np.sin(q5) - np.cos(q2)*np.cos(q4)*np.cos(q5)
        J61 = 0
        J62 = -np.sin(q1)
        J63 = 0
        J64 = -np.cos(q1)*np.cos(q2)
        J65 = - np.cos(q4)*np.sin(q1) - np.cos(q1)*np.sin(q2)*np.sin(q4)
        J66 = np.cos(q1)*np.cos(q2)*np.sin(q5) - np.cos(q5)*np.sin(q1)*np.sin(q4) + np.cos(q1)*np.cos(q4)*np.cos(q5)*np.sin(q2)
        return [[J11, J12, J13, J14, J15, J16],
                [J21, J22, J23, J24, J25, J26],
                [J31, J32, J33, J34, J35, J36],
                [J41, J42, J43, J44, J45, J46],
                [J51, J52, J53, J54, J55, J56],
                [J61, J62, J63, J64, J65, J66]]