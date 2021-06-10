import numpy as np
import FLSpegtransfer.utils.CmnUtil as U
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from scipy.spatial.transform import Rotation
import time


class dvrkKinematics():
    @classmethod
    def pose_to_transform(cls, pos, quat):
        """

        :param pos: position (m)
        :param rot: quaternion (qx, qy, qz, qw)
        :return:
        """
        R = U.quaternion_to_R(quat)
        if np.shape(R) == (3,3):
            T = np.zeros((4, 4))
            T[:3, :3] = R
            T[:3, -1] = pos
            T[-1, -1] = 1
        else:
            T = np.zeros((len(R), 4, 4))
            T[:, :3, :3] = R
            T[:, :3, -1] = pos
            T[:, -1, -1] = 1
        return T

    @classmethod
    def joint_to_pose(cls, joint):
        if joint==[]:
            pos = []
            rot = []
        else:
            T = dvrkKinematics.fk(joints=joint, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=dvrkVar.L4)
            pos = T[:3,-1]
            R = T[:3,:3]
            rot = U.R_to_quaternion(R)
        return pos, rot

    @classmethod
    def pose_to_joint(cls, pos, rot, method='analytic'):
        if pos==[] or rot==[]:
            joint = []
        else:
            T = dvrkKinematics.pose_to_transform(pos, rot)    # current transformation
            joint = dvrkKinematics.ik(T, method)
        return joint

    @classmethod
    def DH_transform(cls, a, alpha, d, theta, unit='rad'):  # modified DH convention
        if unit == 'deg':
            alpha = np.deg2rad(alpha)
            theta = np.deg2rad(theta)
        variables = [a, alpha, d, theta]
        N = 0
        for var in variables:
            if type(var) == np.ndarray or type(var) == list:
                N = len(var)
        if N == 0:
            T = np.array([[np.cos(theta), -np.sin(theta), 0, a],
                          [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha),
                           -np.sin(alpha) * d],
                          [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                          [0, 0, 0, 1]])
        else:
            T = np.zeros((N, 4, 4))
            T[:, 0, 0] = np.cos(theta)
            T[:, 0, 1] = -np.sin(theta)
            T[:, 0, 2] = 0.0
            T[:, 0, 3] = a
            T[:, 1, 0] = np.sin(theta) * np.cos(alpha)
            T[:, 1, 1] = np.cos(theta) * np.cos(alpha)
            T[:, 1, 2] = -np.sin(alpha)
            T[:, 1, 3] = -np.sin(alpha)*d
            T[:, 2, 0] = np.sin(theta) * np.sin(alpha)
            T[:, 2, 1] = np.cos(theta) * np.sin(alpha)
            T[:, 2, 2] = np.cos(alpha)
            T[:, 2, 3] = np.cos(alpha)*d
            T[:, 3, 0] = 0.0
            T[:, 3, 1] = 0.0
            T[:, 3, 2] = 0.0
            T[:, 3, 3] = 1.0
        return T

    @classmethod
    def fk(cls, joints, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=dvrkVar.L4):
        q1, q2, q3, q4, q5, q6 = np.array(joints).T
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

    @classmethod
    def ik(self, T, method='analytic'):
        if method=='analytic':
            T = np.linalg.inv(T)
            if np.shape(T) == (4,4):
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
                joint = [[q1, q2, q3, q4, q5, q6]]
            else:
                x84 = T[:, 0, 3]
                y84 = T[:, 1, 3]
                z84 = T[:, 2, 3]
                q6 = np.arctan2(x84, z84 - dvrkVar.L4)
                temp = -dvrkVar.L3 + np.sqrt(x84 ** 2 + (z84 - dvrkVar.L4) ** 2)
                q3 = dvrkVar.L1 - dvrkVar.L2 + np.sqrt(y84 ** 2 + temp ** 2)
                q5 = np.arctan2(-y84, temp)
                R84 = np.zeros((len(T), 3, 3))
                R84[:, 0, 0] = np.sin(q5)*np.sin(q6)
                R84[:, 0, 1] = -np.cos(q6)
                R84[:, 0, 2] = np.cos(q5) * np.sin(q6)
                R84[:, 1, 0] = np.cos(q5)
                R84[:, 1, 1] = 0.0
                R84[:, 1, 2] = -np.sin(q5)
                R84[:, 2, 0] = np.cos(q6) * np.sin(q5)
                R84[:, 2, 1] = np.sin(q6)
                R84[:, 2, 2] = np.cos(q5) * np.cos(q6)
                # R84 = np.array([[np.sin(q5) * np.sin(q6), -np.cos(q6), np.cos(q5) * np.sin(q6)],
                #                 [np.cos(q5), 0, -np.sin(q5)],
                #                 [np.cos(q6) * np.sin(q5), np.sin(q6), np.cos(q5) * np.cos(q6)]])
                R80 = T[:, :3, :3]
                R40 = np.matmul(R84.transpose(0, 2, 1), R80)
                n32 = R40[:, 2, 1]
                n31 = R40[:, 2, 0]
                n33 = R40[:, 2, 2]
                n22 = R40[:, 1, 1]
                n12 = R40[:, 0, 1]
                q2 = np.arcsin(n32)
                q1 = np.arctan2(-n31, n33)
                q4 = np.arctan2(n22, n12)
                joint = np.array([q1, q2, q3, q4, q5, q6]).T
        # elif method=='numerical':
        #     q0 = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # initial guess
        #     ik_sol = self.ikine(T, q0)
        #     joint = [ik_sol[0, 0], ik_sol[0, 1], ik_sol[0, 2], ik_sol[0, 3], ik_sol[0, 4], ik_sol[0, 5]]
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
    def T04_to_q1234(cls, T04):
        x04 = T04[0, 3]
        y04 = T04[1, 3]
        z04 = T04[2, 3]
        q1 = np.arctan2(x04, -z04)  # (rad)
        q2 = np.arctan2(-y04, np.sqrt(x04 ** 2 + z04 ** 2))  # (rad)
        q3 = np.sqrt(x04 ** 2 + y04 ** 2 + z04 ** 2) + dvrkVar.L1 - dvrkVar.L2  # (m)

        T01 = dvrkKinematics.DH_transform(0, np.pi / 2, 0, q1 + np.pi / 2)
        T12 = dvrkKinematics.DH_transform(0, -np.pi / 2, 0, q2 - np.pi / 2)
        T23 = dvrkKinematics.DH_transform(0, np.pi / 2, q3 - dvrkVar.L1 + dvrkVar.L2, 0)
        T03 = T01.dot(T12).dot(T23)
        R34 = (T03[:3, :3]).T.dot(T04[:3, :3])
        euler = Rotation.from_matrix(R34).as_euler('zxy')
        q4 = euler[0]
        return [q1, q2, q3, q4]

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
        q1,q2,q3,q4,q5,q6 = np.array(joints).T
        J = np.zeros((len(joints), 6, 6))
        J[:, 0, 0] = L2*np.cos(q1)*np.cos(q2) - L1*np.cos(q1)*np.cos(q2) + q3*np.cos(q1)*np.cos(q2) + L3*np.cos(q1)*np.cos(q2)*np.cos(q5) - L4*np.cos(q4)*np.sin(q1)*np.sin(q6) + L3*np.sin(q1)*np.sin(q4)*np.sin(q5) - L3*np.cos(q1)*np.cos(q4)*np.sin(q2)*np.sin(q5) - L4*np.cos(q1)*np.sin(q2)*np.sin(q4)*np.sin(q6) + L4*np.cos(q6)*np.sin(q1)*np.sin(q4)*np.sin(q5) + L4*np.cos(q1)*np.cos(q2)*np.cos(q5)*np.cos(q6) - L4*np.cos(q1)*np.cos(q4)*np.cos(q6)*np.sin(q2)*np.sin(q5)
        J[:, 0, 1] = -np.sin(q1)*(L2*np.sin(q2) - L1*np.sin(q2) + q3*np.sin(q2) + L3*np.cos(q5)*np.sin(q2) + L3*np.cos(q2)*np.cos(q4)*np.sin(q5) + L4*np.cos(q5)*np.cos(q6)*np.sin(q2) + L4*np.cos(q2)*np.sin(q4)*np.sin(q6) + L4*np.cos(q2)*np.cos(q4)*np.cos(q6)*np.sin(q5))
        J[:, 0, 2] = np.cos(q2)*np.sin(q1)
        J[:, 0, 3] = L3*np.sin(q1)*np.sin(q2)*np.sin(q4)*np.sin(q5) - L4*np.cos(q1)*np.sin(q4)*np.sin(q6) - L4*np.cos(q1)*np.cos(q4)*np.cos(q6)*np.sin(q5) - L4*np.cos(q4)*np.sin(q1)*np.sin(q2)*np.sin(q6) - L3*np.cos(q1)*np.cos(q4)*np.sin(q5) + L4*np.cos(q6)*np.sin(q1)*np.sin(q2)*np.sin(q4)*np.sin(q5)
        J[:, 0, 4] = -(L3 + L4*np.cos(q6))*(np.cos(q1)*np.cos(q5)*np.sin(q4) + np.cos(q2)*np.sin(q1)*np.sin(q5) + np.cos(q4)*np.cos(q5)*np.sin(q1)*np.sin(q2))
        J[:, 0, 5] = L4*np.cos(q1)*np.cos(q4)*np.cos(q6) - L4*np.cos(q2)*np.cos(q5)*np.sin(q1)*np.sin(q6) - L4*np.cos(q6)*np.sin(q1)*np.sin(q2)*np.sin(q4) + L4*np.cos(q1)*np.sin(q4)*np.sin(q5)*np.sin(q6) + L4*np.cos(q4)*np.sin(q1)*np.sin(q2)*np.sin(q5)*np.sin(q6)
        J[:, 1, 0] = np.zeros_like(len(joints))
        J[:, 1, 1] = L1*np.cos(q2) - L2*np.cos(q2) - q3*np.cos(q2) - L3*np.cos(q2)*np.cos(q5) - L4*np.cos(q2)*np.cos(q5)*np.cos(q6) + L3*np.cos(q4)*np.sin(q2)*np.sin(q5) + L4*np.sin(q2)*np.sin(q4)*np.sin(q6) + L4*np.cos(q4)*np.cos(q6)*np.sin(q2)*np.sin(q5)
        J[:, 1, 2] = -np.sin(q2)
        J[:, 1, 3] = np.cos(q2)*(L3*np.sin(q4)*np.sin(q5) - L4*np.cos(q4)*np.sin(q6) + L4*np.cos(q6)*np.sin(q4)*np.sin(q5))
        J[:, 1, 4] = (L3 + L4*np.cos(q6))*(np.sin(q2)*np.sin(q5) - np.cos(q2)*np.cos(q4)*np.cos(q5))
        J[:, 1, 5] = L4*np.cos(q5)*np.sin(q2)*np.sin(q6) - L4*np.cos(q2)*np.cos(q6)*np.sin(q4) + L4*np.cos(q2)*np.cos(q4)*np.sin(q5)*np.sin(q6)
        J[:, 2, 0] = L2*np.cos(q2)*np.sin(q1) - L1*np.cos(q2)*np.sin(q1) + q3*np.cos(q2)*np.sin(q1) + L3*np.cos(q2)*np.cos(q5)*np.sin(q1) + L4*np.cos(q1)*np.cos(q4)*np.sin(q6) - L3*np.cos(q1)*np.sin(q4)*np.sin(q5) + L4*np.cos(q2)*np.cos(q5)*np.cos(q6)*np.sin(q1) - L4*np.cos(q1)*np.cos(q6)*np.sin(q4)*np.sin(q5) - L3*np.cos(q4)*np.sin(q1)*np.sin(q2)*np.sin(q5) - L4*np.sin(q1)*np.sin(q2)*np.sin(q4)*np.sin(q6) - L4*np.cos(q4)*np.cos(q6)*np.sin(q1)*np.sin(q2)*np.sin(q5)
        J[:, 2, 1] = np.cos(q1)*(L2*np.sin(q2) - L1*np.sin(q2) + q3*np.sin(q2) + L3*np.cos(q5)*np.sin(q2) + L3*np.cos(q2)*np.cos(q4)*np.sin(q5) + L4*np.cos(q5)*np.cos(q6)*np.sin(q2) + L4*np.cos(q2)*np.sin(q4)*np.sin(q6) + L4*np.cos(q2)*np.cos(q4)*np.cos(q6)*np.sin(q5))
        J[:, 2, 2] = -np.cos(q1)*np.cos(q2)
        J[:, 2, 3] = L4*np.cos(q1)*np.cos(q4)*np.sin(q2)*np.sin(q6) - L4*np.sin(q1)*np.sin(q4)*np.sin(q6) - L3*np.cos(q4)*np.sin(q1)*np.sin(q5) - L4*np.cos(q4)*np.cos(q6)*np.sin(q1)*np.sin(q5) - L3*np.cos(q1)*np.sin(q2)*np.sin(q4)*np.sin(q5) - L4*np.cos(q1)*np.cos(q6)*np.sin(q2)*np.sin(q4)*np.sin(q5)
        J[:, 2, 4] = (L3 + L4*np.cos(q6))*(np.cos(q1)*np.cos(q2)*np.sin(q5) - np.cos(q5)*np.sin(q1)*np.sin(q4) + np.cos(q1)*np.cos(q4)*np.cos(q5)*np.sin(q2))
        J[:, 2, 5] = L4*np.cos(q4)*np.cos(q6)*np.sin(q1) + L4*np.cos(q1)*np.cos(q2)*np.cos(q5)*np.sin(q6) + L4*np.cos(q1)*np.cos(q6)*np.sin(q2)*np.sin(q4) + L4*np.sin(q1)*np.sin(q4)*np.sin(q5)*np.sin(q6) - L4*np.cos(q1)*np.cos(q4)*np.sin(q2)*np.sin(q5)*np.sin(q6)
        J[:, 3, 0] = np.zeros_like(len(joints))
        J[:, 3, 1] = -np.cos(q1)
        J[:, 3, 2] = np.zeros_like(len(joints))
        J[:, 3, 3] = np.cos(q2)*np.sin(q1)
        J[:, 3, 4] = np.sin(q1)*np.sin(q2)*np.sin(q4) - np.cos(q1)*np.cos(q4)
        J[:, 3, 5] = -np.cos(q1)*np.cos(q5)*np.sin(q4) - np.cos(q2)*np.sin(q1)*np.sin(q5) - np.cos(q4)*np.cos(q5)*np.sin(q1)*np.sin(q2)
        J[:, 4, 0] = -np.ones_like(len(joints))
        J[:, 4, 1] = np.zeros_like(len(joints))
        J[:, 4, 2] = np.zeros_like(len(joints))
        J[:, 4, 3] = -np.sin(q2)
        J[:, 4, 4] = np.cos(q2)*np.sin(q4)
        J[:, 4, 5] = np.sin(q2)*np.sin(q5) - np.cos(q2)*np.cos(q4)*np.cos(q5)
        J[:, 5, 0] = np.zeros_like(len(joints))
        J[:, 5, 1] = -np.sin(q1)
        J[:, 5, 2] = np.zeros_like(len(joints))
        J[:, 5, 3] = -np.cos(q1)*np.cos(q2)
        J[:, 5, 4] = -np.cos(q4)*np.sin(q1) - np.cos(q1)*np.sin(q2)*np.sin(q4)
        J[:, 5, 5] = np.cos(q1)*np.cos(q2)*np.sin(q5) - np.cos(q5)*np.sin(q1)*np.sin(q4) + np.cos(q1)*np.cos(q4)*np.cos(q5)*np.sin(q2)
        return J
