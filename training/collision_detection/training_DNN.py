from FLSpegtransfer.training.dvrkCurrEstDNN import NNModel, dvrkCurrEstDNN
from FLSpegtransfer.utils.Filter.filters import LPF
from FLSpegtransfer.utils.plot import *

# Load training & verification dataset
root = "/home/hwangmh/pycharmprojects/FLSpegtransfer/"
dir = "training/collision_detection/dataset/random_sampled/low_acc/"
joints_des = np.load(root + dir + "joint_des.npy", allow_pickle=True, encoding="latin1")
joints_msd = np.load(root + dir + "joint_msd.npy", allow_pickle=True, encoding="latin1")
mot_curr = np.load(root + dir + "mot_curr.npy", allow_pickle=True, encoding="latin1")

joints_des = np.concatenate(joints_des, axis=0)
joints_msd = np.concatenate(joints_msd, axis=0)
mot_curr = np.concatenate(mot_curr, axis=0)

len_train = int(len(joints_des)*0.9)
joints_tr = joints_des[:len_train, :3]
joints_ver = joints_des[len_train:, :3]
mot_curr_tr = mot_curr[:len_train, :3]
mot_curr_ver = mot_curr[len_train:, :3]
print ("total dataset:", len(joints_des))
print ("training dataset:", len_train)


# Load models for verification
nb_axis = 3
nb_ensemble = 10
history = 50
training = True
nb_epoch = 3000
verification = False
LPF = LPF(fc=3, fs=100, order=2, nb_axis=nb_axis)

if training:
    for i in range(nb_ensemble):
        # Input dataset
        q = joints_tr

        # Output dataset
        mot_curr_raw = mot_curr_tr
        mot_curr = LPF.filter(mot_curr_raw)
        plot_filter(mot_curr_raw[50:3000, :nb_axis], mot_curr[50:3000, :nb_axis], show_window=True)

        # train model
        x = q
        y = mot_curr
        input_dim = np.shape(x)[1]*history
        output_dim = np.shape(y)[1]
        model_tr = NNModel(input_dim, output_dim)
        model_tr.train(data_in=x, data_out=y, batch_size=1000, history=history, num_workers=3, num_epoch=nb_epoch)

        # training result
        model_tr.save('model_mot_curr_est.out' + str(i))


if verification:
    # load trained models
    model_ver = dvrkCurrEstDNN(history=history, nb_ensemble=nb_ensemble, nb_axis=nb_axis)

    # Input dataset
    q = joints_ver[500:, :nb_axis]

    # Output dataset
    mot_curr_raw = mot_curr_ver[500:, :nb_axis]

    # prediction from model
    y_ver = []
    y_est = []
    y_comp = []
    for i, (joint, curr_raw) in enumerate(zip(q, mot_curr_raw)):
        curr_filtered = LPF.filter([curr_raw])
        y_ver.append(np.squeeze(curr_filtered))
        y_est.append(model_ver.predict(joint))
        print (i)

    # verification result
    y_est = np.array(y_est)
    y_ver = np.array(y_ver)
    t = np.array(range(len(y_ver)))*0.01  # (ms)
    for i in range(nb_axis):
        plt.subplot(str(nb_axis)+"1"+str(i+1))
        plt.plot(t[500:6000], y_ver[500:6000, i], 'b-', t[500:6000], y_est[500:6000, i], 'r-', t[500:6000], (y_ver - y_est)[500:6000, i], 'g-')
        plt.ylabel('mot_curr'+str(i+1)+' (mA)')
    plt.xlabel('time (s)')
    plt.legend(['measured', 'predicted', 'compensated'], ncol=3, bbox_to_anchor=(0.5, 3.5), loc="lower center")
    plt.show()

    print("Maximum current (A)")
    print("Orig. =", np.max(abs(y_ver[500:]), axis=0))
    print("Comp. =", np.max(abs((y_ver - y_est)[500:]), axis=0))
