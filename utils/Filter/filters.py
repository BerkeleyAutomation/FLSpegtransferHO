from scipy.signal import butter, filtfilt, lfilter, lfilter_zi, firwin, iirnotch
import numpy as np


class FIRWIN:
    def __init__(self, fc, fs, nb_axis=1): # fc=cut off freq, fs=sampling rate
        nyq = fs // 2
        norm_cutoff = fc / nyq
        numtaps = 20    # Length of the filter
        self.nb_axis = nb_axis
        self.b = firwin(numtaps, norm_cutoff)
        # self.b = firwin(numtaps, fc, nyq=nyq)
        self.z = [lfilter_zi(self.b, 1) for _ in range(nb_axis)]

    def filter(self, data):   # online filter
        data = np.array(data)
        result = [lfilter(self.b, 1, data[:, i], zi=self.z[i]) for i in range(self.nb_axis)]
        data_filtered = []
        for i, (data, z) in enumerate(result):
            data_filtered.append(data)
            self.z[i] = z
        return np.array(data_filtered).T


class LPF:
    def __init__(self, fc, fs, order, nb_axis=1): # fc=cut off freq, fs=sampling rate
        self.nb_axis = nb_axis
        nyq = fs / 2.0
        normal_cutoff = fc / nyq
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)    # Get the filter coefficients
        self.z = [lfilter_zi(self.b, self.a) for _ in range(self.nb_axis)]
        self.nb_axis = nb_axis

    def filter(self, data):   # online filter
        data = np.array(data).reshape(-1, self.nb_axis)
        result = [lfilter(self.b, self.a, data[:, i], zi=self.z[i]) for i in range(self.nb_axis)]
        data_filtered = []
        for i, (data, z) in enumerate(result):
            data_filtered.append(data)
            self.z[i] = z
        return np.array(data_filtered).T

class Notch:
    def __init__(self, f0, fs, Q, nb_axis=1):
        self.nb_axis = nb_axis
        # f0 = freq to be removed
        # fs = sampling rate (Hz)
        # Q = quality factor
        self.b, self.a = iirnotch(f0, Q, fs)    # Get the filter coefficients
        self.z = [lfilter_zi(self.b, self.a) for _ in range(nb_axis)]

    def filter(self, data):  # online filter
        data = np.array(data)
        result = [lfilter(self.b, self.a, data[:, i], zi=self.z[i]) for i in range(self.nb_axis)]
        data_filtered = []
        for i, (data, z) in enumerate(result):
            data_filtered.append(data)
            self.z[i] = z
        return np.array(data_filtered).T


# Required input defintions are as follows;
# time:   Time between samples
# band:   The bandwidth around the centerline freqency that you wish to filter
# freq:   The centerline frequency to be filtered
# ripple: The maximum passband ripple that is allowed in db
# order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
#         IIR filters are best suited for high values of order.  This algorithm
#         is hard coded to FIR filters
# filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
# data:         the data to be filtered
def Implement_Notch_Filter(time, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter
    fs = 1 / time
    nyq = fs / 2.0
    low = freq - band / 2.0
    high = freq + band / 2.0
    low = low / nyq
    high = high / nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop', analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs//2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# calculate velocity & acceleration
def vel_acc_profile(q, time_step):      # naive differentiation
    # velocity traj.
    q_prev = np.insert(q, 0, 0, axis=0)
    q_prev = np.delete(q_prev, -1, axis=0)
    qv = (q - q_prev)/time_step
    qv[0] = 0.0

    # acceleration traj.
    qv_prev = np.insert(qv, 0, 0, axis=0)
    qv_prev = np.delete(qv_prev, -1, axis=0)
    qa = (qv - qv_prev)/time_step
    qa[0] = 0.0
    return qv, qa


if __name__ == '__main__':
    import FLSpegtransfer.utils.plot as plot

    root = "/home/hwangmh/pycharmprojects/FLSpegtransfer/"
    dir = "training/collision_detection/dataset/random_sampled/"
    joints = np.load(root + dir + "joint.npy")
    mot_curr = np.load(root + dir + "mot_curr.npy")

    data = mot_curr[:2000,0]

    f = LPF(fc=5, fs=100)

    # result = f.filter(data)
    result = np.zeros(data.size)
    for i, x in enumerate(data):
        result[i] = f.filter([x])

    data_filtered = result
    plot.plot_filter(data[50:], data_filtered[50:], show_window=True)