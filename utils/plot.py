import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')
# plt.style.use('bmh')
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=17)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title


def plot_filter(data, data_filtered, show_window=True):
    data = np.array(data)
    data_filtered = np.array(data_filtered)

    # Create plot
    ax = plt.subplot(211)
    plt.plot(data, 'b-')
    # plt.xticks(np.arange(0, 1000, step=20))
    plt.ylabel('raw')

    ax = plt.subplot(212)
    plt.plot(data_filtered, 'b-')
    plt.ylabel('filtered')
    plt.xlabel('(samples)')
    if show_window:
        plt.show()


def plot_trajectory(q, nb_axis, show_window=True):
    q = np.array(q)

    plt.figure(1)
    for i in range(nb_axis):
        plt.subplot(str(nb_axis)+"1"+str(i+1))
        plt.plot(q[i,:], 'b-')

    if show_window:
        plt.show()