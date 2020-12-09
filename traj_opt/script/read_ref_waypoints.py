import numpy as np

waypoints = np.load("ref_waypoints.npy")
for wp in waypoints:
    q0 = wp[0]
    qw1 = wp[1]
    qw2 = wp[2]
    qf = wp[3]
    print ("starting conf: ", q0)
    print("waypoint1: ", qw1)
    print("waypoint2: ", qw2)
    print("final conf: ", qf)
