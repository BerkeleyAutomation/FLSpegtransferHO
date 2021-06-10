import numpy as np

# dvrk variables
L1 = 0.4318  # Rcc (m)
L2 = 0.4162  # tool
# L2 = 0.4662  # longer tool
L3 = 0.0091  # pitch ~ yaw (m)
L4 = 0.0095  # yaw ~ tip (m)
vel_ratio = 2.0
acc_ratio = 2.0
v_max = np.array([np.pi, np.pi, 0.2, 3*2*np.pi, 3*2*np.pi, 3*2*np.pi])*vel_ratio       # max velocity (rad/s) or (m/s)
a_max = np.array([np.pi, np.pi, 0.2, 2*2*np.pi, 2*2*np.pi, 2*2*np.pi])*acc_ratio       # max acceleration (rad/s^2) or (m/s^2)

# mJointTrajectory.VelocityMaximum.Ref(2, 0).SetAll(180.0 * cmnPI_180); // degrees per second
# mJointTrajectory.VelocityMaximum.Element(2) = 0.2; // m per second
# mJointTrajectory.VelocityMaximum.Ref(4, 3).SetAll(3.0 * 360.0 * cmnPI_180);
# SetJointVelocityRatio(1.0);
# mJointTrajectory.AccelerationMaximum.Ref(2, 0).SetAll(180.0 * cmnPI_180);
# mJointTrajectory.AccelerationMaximum.Element(2) = 0.2; // m per second
# mJointTrajectory.AccelerationMaximum.Ref(4, 3).SetAll(2.0 * 360.0 * cmnPI_180);
# SetJointAccelerationRatio(1.0);
