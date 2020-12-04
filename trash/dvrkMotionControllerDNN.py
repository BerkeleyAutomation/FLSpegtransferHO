import numpy as np

from FLSpegtransfer.training.dvrkHystCompDNN import NNModel


class dvrkNNController():
    def __init__(self, filename, nb_ensemble):
        self.use_history = True
        self.H = 6

        # load model
        self.nb_ensemble = nb_ensemble
        self.models = []
        for i in range(self.nb_ensemble):
            model = NNModel(input_dim=18, output_dim=3)
            model.load_model(filename+str(i))
            self.models.append(model)

        # data members
        self.q_cmd = np.array([0.0, 0.0, 0.0])
        self.q_cmd_prev = np.array([0.0, 0.0, 0.0])
        self.q_cmd_hist = []

        # for hysteresis model
        self.w_cmd = np.array([0.0, 0.0, 0.0])
        self.w_cmd_prev = np.array([0.0, 0.0, 0.0])
        self.H_upper = [np.deg2rad(-7), np.deg2rad(17), np.deg2rad(2)]
        self.H_lower = [np.deg2rad(-13), np.deg2rad(-10), np.deg2rad(-30)]
        self.set_point = [0.0, 0.0, 0.0]
        self.track = [0, 0, 0]
        self.margin_L = [0.0, 0.0, 0.0]
        self.margin_R = [0.0, 0.0, 0.0]

        # data member for interpolation
        self.v_max = 2.0  # maximum velocity
        self.dt = 0.01  # sampling time interval
        self.q_cmd_int_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def model_out(self, x):
        preds = []
        for model in self.models:
            preds.append(model.model_out(x))
        return np.median(preds, axis=0)

    def predict_using_history(self, q_cmd):
        assert self.use_history
        if len(self.q_cmd_hist) < self.H*3:
            self.q_cmd_hist = np.insert(self.q_cmd_hist, 0, q_cmd[3:])
            q_pred = np.concatenate((q_cmd[:3], [0.0 ,0.0, 0.0]))
        else:
            self.q_cmd_hist = np.insert(self.q_cmd_hist, 0, q_cmd[3:])
            self.q_cmd_hist = np.delete(self.q_cmd_hist, np.s_[-3:])
            q_pred = self.model_out(self.q_cmd_hist)
            q_pred = np.concatenate((q_cmd[:3], q_pred))
        return q_pred

    # predict q_act
    def predict(self, q_cmd):
        self.q_cmd_prev = self.q_cmd
        self.q_cmd = q_cmd[3:]      # using wrist joints
        self.w_cmd_prev = self.w_cmd
        self.w_cmd = np.array(self.q_cmd) - np.array(self.q_cmd_prev)

        set_points = []
        tracks = []
        marginsL = []
        marginsR = []
        for i, q in enumerate(self.q_cmd):
            if self.set_point[i] + self.H_lower[i] < q < self.set_point[i] + self.H_upper[i]:
                self.track[i] = 0
                self.margin_L[i] = q - (self.set_point[i] + self.H_lower[i])
                self.margin_R[i] = (self.set_point[i] + self.H_upper[i]) - q
            elif self.set_point[i] + self.H_upper[i] <= q:
                self.track[i] = 1
                self.set_point[i] = q - self.H_upper[i]
                self.margin_L[i] = self.H_upper[i] - self.H_lower[i]
                self.margin_R[i] = 0.0
            elif q <= self.set_point[i] + self.H_lower[i]:
                self.track[i] = -1
                self.set_point[i] = q - self.H_lower[i]
                self.margin_L[i] = 0.0
                self.margin_R[i] = self.H_upper[i] - self.H_lower[i]

            set_points.append(self.set_point[i])
            tracks.append(self.track[i])
            marginsL.append(self.margin_L[i])
            marginsR.append(self.margin_R[i])

        x = list(self.q_cmd) + list(self.w_cmd) + set_points + marginsL + marginsR
        q_pred = self.model_out(x)
        q_pred = np.concatenate((q_cmd[:3], q_pred))
        return q_pred

    # motion controller (calibrate new cmd)
    def step(self, q_cmd):
        assert q_cmd != []
        if self.use_history:
            q_pred = self.predict_using_history(q_cmd)
        else:
            q_pred = self.predict(q_cmd)
        error = np.array(q_cmd) - np.array(q_pred)
        q_cmd_new = np.array(q_cmd) + error
        return q_cmd_new

    # predict & calibrate high_acc points
    def cal_interpolate(self, q_cmd, mode):
        assert q_cmd != []
        assert not self.use_history
        if all(np.isclose(q_cmd, self.q_cmd_int_prev)):
            return q_cmd
        else:
            q_cmd = np.array(q_cmd)
            tf = abs((q_cmd - self.q_cmd_int_prev) / self.v_max)
            n = np.ceil(max(tf / self.dt))  # number of points high_acc
            dq = (q_cmd - self.q_cmd_int_prev) / n
            for i in range(int(n)):
                if mode=='predict':
                    q_cmd_new = self.predict(self.q_cmd_int_prev + (i+1)*dq)    # predict
                elif mode=='calibrate':
                    q_cmd_new = self.step(self.q_cmd_int_prev + (i+1)*dq)    # calibrate
            self.q_cmd_int_prev = q_cmd
            return q_cmd_new    # return the last one