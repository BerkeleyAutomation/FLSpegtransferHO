import sympy as sym
from sympy import sin, cos, simplify, pprint


class dVRKKinematicsSym:
    def __init__(self):
        L1, L2, L3 = sym.symbols('L1, L2, L3')
        q1, q2, q3, q4, q5, q6, jaw = sym.symbols('q1, q2, q3, q4, q5, q6, jaw')
        T01 = self.DH_transform(0, sym.pi / 2, 0, q1 + sym.pi / 2)
        T12 = self.DH_transform(0, -sym.pi / 2, 0, q2 - sym.pi / 2)
        T23 = self.DH_transform(0, sym.pi / 2, q3 - L1 + L2, 0)
        T34 = self.DH_transform(0, 0, 0, q4)
        T03 = T01 * T12 * T23
        T04 = T01 * T12 * T23 * T34
        import pdb; pdb.set_trace()
        print (T04)

    @classmethod
    def DH_transform(cls, a, alpha, d, theta):
        T = sym.Matrix([[cos(theta), -sin(theta), 0, a],
                        [sin(theta) * cos(alpha), cos(theta) * cos(alpha), -sin(alpha), -sin(alpha) * d],
                        [sin(theta) * sin(alpha), cos(theta) * sin(alpha), cos(alpha), cos(alpha) * d],
                        [0, 0, 0, 1]])
        return T

    @classmethod
    def get_keypoints_model_sym(cls):   # symbolic calculation to reduce calculation time and to formulate opt. eqn.
        L1, L2, L3 = sym.symbols('L1, L2, L3')
        q1, q2, q3, q4, q5, q6, jaw = sym.symbols('q1, q2, q3, q4, q5, q6, jaw')

        T01 = cls.DH_transform(0, sym.pi / 2, 0, q1 + sym.pi / 2)
        T12 = cls.DH_transform(0, -sym.pi / 2, 0, q2 - sym.pi / 2)
        T23 = cls.DH_transform(0, sym.pi / 2, q3 - L1 + L2, 0)
        T34 = cls.DH_transform(0, 0, 0, q4)
        T45 = cls.DH_transform(0, -sym.pi / 2, 0, q5 - sym.pi / 2)
        T56_tip1 = cls.DH_transform(L3, -sym.pi / 2, 0, q6 - jaw / 2 - sym.pi / 2)
        T56_tip2 = cls.DH_transform(L3, -sym.pi / 2, 0, q6 + jaw / 2 - sym.pi / 2)
        T04 = T01 * T12 * T23 * T34
        T05 = T04 * T45
        T06_tip1 = T05 * T56_tip1
        T06_tip2 = T05 * T56_tip2


if __name__ == "__main__":
    kin = dVRKKinematicsSym()