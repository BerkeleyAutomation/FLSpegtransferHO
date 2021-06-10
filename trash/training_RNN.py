from FLSpegtransfer.trash.RNNModel import RNNModel
from scipy.signal import butter, filtfilt
from FLSpegtransfer.utils.plot import *

# Load training & verification dataset
root = "/home/hwangmh/pycharmprojects/FLSpegtransfer/"
dir = "training/collision_detection/dataset/random_sampled/"
joints = np.load(root + dir + "joint.npy")
mot_curr = np.load(root + dir + "mot_curr.npy")

len_train = int(len(joints)*0.9)
joints_tr = joints[:len_train, :3]
joints_ver = joints[len_train:, :3]
mot_curr_tr = mot_curr[:len_train, :3]
mot_curr_ver = mot_curr[len_train:, :3]
print ("total dataset:", len(joints))
print ("training dataset:", len_train)

nb_ensemble = 10
history = 100
training = True
nb_epoch = 500
verification = False

if training:
    for i in range(nb_ensemble):
        # define joint angle
        q_raw = joints_tr
        # Filter the data, and plot both the original and filtered signals.
        # q = butter_lowpass_filter(q_raw, cutoff=2, fs=100, order=2)
        # plot_filter(q_raw, q, show_window=True)
        q = q_raw

        mot_curr_raw = mot_curr_tr
        # mot_curr = butter_lowpass_filter(mot_curr_raw, cutoff=2, fs=100, order=2)
        # plot_filter(mot_curr_raw, mot_curr, show_window=True)
        mot_curr = mot_curr_raw


        # # define velocity, acceleration
        # qdt_raw, qdtdt_raw = vel_acc_profile(q, time_step=0.01)
        # qdt = qdt_raw
        # # qdt = butter_lowpass_filter(qdt_raw, cutoff=20, fs=100, order=2)
        # # plot_filter(qdt_raw, qdt, show_window=True)
        # qdtdt = butter_lowpass_filter(qdtdt_raw, cutoff=2, fs=100, order=2)
        # # plot_filter(qdtdt_raw, qdtdt, show_window=True)

        # # plot training dataset
        # data = np.concatenate((q, qdt, qdtdt, mot_curr)).reshape(4,-1)[:,:3000]
        # # plot_trajectory(data, nb_axis=4, show_window=True)


        # training dataset
        # q = q.reshape(-1,1)
        # mot_curr = mot_curr.reshape(-1,1)
        # x, y = format_dataset(q, mot_curr, H=history)


        # train model
        x = q[:3000]
        y = mot_curr[:3000]
        input_dim = np.shape(x)[1]
        output_dim = np.shape(y)[1]
        model_tr = RNNModel(input_dim, output_dim, num_layers=1)
        model_tr.train(data_in=x, data_out=y, history=history, num_epoch=nb_epoch)

        x_ver, y_ver = model_tr.format_dataset(x, y, history=history)
        y_pred = []
        for input in x_ver:
            output = model_tr.model_out(input)
            y_pred.append(output)
        y_ver = np.array(y_ver)
        y_ver = np.squeeze(y_ver).reshape(-1,3)
        y_pred = np.squeeze(y_pred).reshape(-1, 3)

        import matplotlib.pyplot as plt

        plt.subplot(311)
        plt.plot(y_ver[:,0], 'b-', y_pred[:,0], 'r-')
        plt.subplot(312)
        plt.plot(y_ver[:, 0], 'b-', y_pred[:, 0], 'r-')
        plt.subplot(313)
        plt.plot(y_ver[:, 0], 'b-', y_pred[:, 0], 'r-')
        plt.show()

        # training result
        model_tr.save('model_mot_curr_est.out'+str(i))


if verification:
    q_raw = joints_ver
    mot_curr_raw = mot_curr_ver

    q = q_raw
    mot_curr = mot_curr_raw

    # prediction from model
    y_est = []
    y_ver = []
    for i, (joint, curr) in enumerate(zip(q, mot_curr)):
        y_est.append(model_ver.predict(joint))
        y_ver.append(curr)
        print (i)

    # verification result
    import matplotlib.pyplot as plt
    # y_est = np.squeeze(y_est)
    # y_ver = np.squeeze(y_ver)
    y_est = np.array(y_est)
    y_ver = np.array(y_ver)
    plt.subplot(311)
    plt.plot(y_ver[:, 0], 'b-', y_est[:, 0], 'r-', (y_ver - y_est)[:, 0], 'g-')
    plt.subplot(312)
    plt.plot(y_ver[:, 1], 'b-', y_est[:, 1], 'r-', (y_ver - y_est)[:, 1], 'g-')
    plt.subplot(313)
    plt.plot(y_ver[:, 2], 'b-', y_est[:, 2], 'r-', (y_ver - y_est)[:, 2], 'g-')
    plt.show()


    # plt.plot(y_ver[100:-100], 'b-', y_est[100:-100], 'r-', (y_ver - y_est)[100:-100], 'g-')
    # plt.show()

    # RMSE = []
    # MSE = []
    # for i in range(6):
    #     RMSE.append(np.sqrt(np.sum((q_des[:,i] - q_act[:,i]) ** 2) / len(q_act[:,i])))
    #     # RMSE.append(np.sqrt(np.sum((q_act[:,i] - q_est[:, i]) ** 2) / len(q_act[:, i])))
    #     MSE.append(np.sum((q_act[:,i] - q_est[:,i]) ** 2) / len(q_act[:,i]))
    #
    # RMSE = [np.rad2deg(q) if i!=2 else q for i,q in enumerate(RMSE)]
    # print("RMSE=", RMSE, "(deg or mm)")
    # print("MSE=", MSE)

    # y_pred = []
    # for input in x:
    #     output = model_NN.model_out(input)
    #     y_pred.append(output)
