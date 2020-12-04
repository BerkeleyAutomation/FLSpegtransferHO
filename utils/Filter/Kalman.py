import numpy as np

class Kalman:
    def __init__(self, A, H, Q, R, x0, P0):
        self.A = np.array(A)
        self.H = np.array(H)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.x = np.array(x0)
        self.P = np.array(P0)

    def estimate(self, z):
        # predict
        x_pred = self.A.dot(self.x)
        P_pred = self.A.dot(self.P).dot(self.A.T) + self.Q

        # update Kalman gain
        K = P_pred.dot(self.H.T).dot(np.linalg.inv(self.H.dot(P_pred).dot(self.H.T) + self.R))

        # estimate
        z = np.array(z)
        self.x = x_pred + K.dot(z - self.H.dot(x_pred))
        self.P = P_pred - K.dot(self.H).dot(P_pred)
        return self.x, self.P