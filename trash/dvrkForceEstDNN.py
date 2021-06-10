import numpy as np
from FLSpegtransfer.training.dvrkHystCompDNN import NNModel


class dvrkForceEstDNN():
    def __init__(self, history, nb_ensemble, nb_axis):

        # model inform.
        self.H = history
        self.nb_axis = nb_axis
        root = "/home/hwangmh/pycharmprojects/FLSpegtransfer/"
        dir = "training/collision_detection/models/DNN/high_acc_3Hz/"
        filename = root + dir + "model_mot_curr_est.out"
        input_dim = self.nb_axis*history
        output_dim = self.nb_axis

        # load model
        self.models = []
        for i in range(nb_ensemble):
            model = NNModel(input_dim, output_dim)
            model.load_model(filename+str(i))
            self.models.append(model)

        # data members
        self.q_cmd_hist = []

    def model_out(self, x):
        preds = []
        for model in self.models:
            preds.append(model.model_out(x))
        return np.median(preds, axis=0)

    def predict(self, q_cmd):
        if len(self.q_cmd_hist) < self.H * self.nb_axis:
            self.q_cmd_hist = np.insert(self.q_cmd_hist, 0, q_cmd)
            q_pred = [0.0]*self.nb_axis
        else:
            self.q_cmd_hist = np.insert(self.q_cmd_hist, 0, q_cmd)
            self.q_cmd_hist = np.delete(self.q_cmd_hist, np.s_[-self.nb_axis:])  # slice
            q_pred = self.model_out(self.q_cmd_hist)
        return q_pred