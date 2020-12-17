"""Shared methods, to be loaded in other code.
"""
import numpy as np

ESC_KEYS = [27, 1048603]
MILLION = float(10**6)


def normalize(v):
    norm=np.linalg.norm(v, ord=2)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def LPF(raw_data, fc, dt):
    filtered = np.zeros_like(raw_data)
    for i in range(len(raw_data)):
        if i==0:
            filtered[0] = raw_data[0]
        else:
            filtered[i] = 2*np.pi*fc*dt*raw_data[i] + (1-2*np.pi*fc*dt)*filtered[i-1]
    return filtered


def euler_to_quaternion(rot, unit='rad'):
    if unit=='deg':
        rot = np.deg2rad(rot)
    # for the various angular functions
    yaw, pitch, roll = rot.T      # yaw (Z), pitch (Y), roll (X)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # quaternion
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr
    return np.array([qx, qy, qz, qw]).T


def quaternion_to_euler(q, unit='rad'):
    qx, qy, qz, qw = np.array(q).T

    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(abs(sinp) >= np.ones_like(sinp), np.sign(sinp)*(np.pi/2), np.arcsin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    if unit=='deg':
        [yaw, pitch, roll] = np.rad2deg([yaw,pitch,roll])
    return np.array([yaw,pitch,roll]).T     # [Z, Y, X]


# def quaternion_to_R(q):
#     qx, qy, qz, qw = q
#     s=np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
#     r11 = 1-2*s*(qy*qy+qz*qz); r12 = 2*s*(qx*qy-qz*qw);  r13 = 2*s*(qx*qz+qy*qw)
#     r21 = 2*s*(qx*qy+qz*qw);  r22 = 1-2*s*(qx*qx+qz*qz); r23 = 2*s*(qy*qz-qx*qw)
#     r31 = 2*s*(qx*qz-qy*qw);  r32 = 2*s*(qy*qz+qx*qw);  r33 = 1-2*s*(qx*qx+qy*qy)
#     R = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
#     return R

def Rx(theta):
    if np.size(theta) == 1:
        return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    else:
        R = np.eye(3)[np.newaxis, :, :]
        R = np.repeat(R, len(theta), axis=0)
        R[:, 1, 1] = np.cos(theta)
        R[:, 1, 2] = -np.sin(theta)
        R[:, 2, 1] = np.sin(theta)
        R[:, 2, 2] = np.cos(theta)
        return R

def Ry(theta):
    if np.size(theta) == 1:
        return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    else:
        R = np.eye(3)[np.newaxis, :, :]
        R = np.repeat(R, len(theta), axis=0)
        R[:, 0, 0] = np.cos(theta)
        R[:, 0, 2] = np.sin(theta)
        R[:, 2, 0] = -np.sin(theta)
        R[:, 2, 2] = np.cos(theta)
        return R

def Rz(theta):
    if np.size(theta) == 1:
        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    else:
        R = np.eye(3)[np.newaxis,:,:]
        R = np.repeat(R, len(theta), axis=0)
        R[:, 0, 0] = np.cos(theta)
        R[:, 0, 1] = -np.sin(theta)
        R[:, 1, 0] = np.sin(theta)
        R[:, 1, 1] = np.cos(theta)
        return R

def euler_to_R(euler_angles):
    theta_z, theta_y, theta_x = euler_angles.T
    Rotz = Rz(theta_z)
    Roty = Ry(theta_y)
    Rotx = Rx(theta_x)
    return np.matmul(np.matmul(Rotz, Roty), Rotx)

# R to ZYX euler angle
def R_to_euler(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([z, y, x])

def R_to_quaternion(R):
    euler = R_to_euler(R)
    return euler_to_quaternion(euler)


def quaternion_to_R(q):
    euler = quaternion_to_euler(q)
    return euler_to_R(euler)

# ZYZ euler angle to quaternion
def inclined_orientation(axis_rot, latitude, longitude=0):
    theta_z1 = longitude
    theta_y = latitude
    theta_z2 = axis_rot
    R = U.Rz(theta_z1).dot(U.Ry(theta_y)).dot(U.Rz(theta_z2))
    return U.R_to_quaternion(R)

def transform(points, T):
    R = T[:3,:3]
    t = T[:3,-1]
    transformed = R.dot(points.T).T + t.T
    return transformed

# Get a rigid transformation matrix from pts1 to pts2
def get_rigid_transform(pts1, pts2):    # Make sure that this is opposite to the coordinate transform
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    mean1 = pts1.mean(axis=0)
    mean2 = pts2.mean(axis=0)
    pts1 = np.array([p - mean1 for p in pts1])
    pts2 = np.array([p - mean2 for p in pts2])
    # if option=='clouds':
    H = pts1.T.dot(pts2)   # covariance matrix
    U,S,V = np.linalg.svd(H)
    V = V.T
    R = V.dot(U.T)
    t = -R.dot(mean1.T) + mean2.T
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, -1] = t
    T[-1, -1] = 1
    return T


def minor(arr,i,j):
    # ith row, jth column removed
    arr = np.array(arr)
    return arr[np.array(list(range(i))+list(range(i+1,arr.shape[0])))[:,np.newaxis],
               np.array(list(range(j))+list(range(j+1,arr.shape[1])))]


def create_waveform(data_range, amp1, amp2, amp3, amp4, freq1, freq2, freq3, freq4, phase, step):
    t = np.arange(0, 1, 1.0 / step)
    waveform1 = amp1*np.sin(2*np.pi*freq1*(t-phase))
    waveform2 = amp2*np.sin(2*np.pi*freq2*(t-phase))
    waveform3 = amp3*np.sin(2*np.pi*freq3*(t-phase))
    waveform4 = amp4*np.sin(2*np.pi*freq4*(t-phase))
    waveform = waveform1 + waveform2 + waveform3 + waveform4
    x = waveform / max(waveform)
    y = (data_range[1]-data_range[0])/2.0*x + (data_range[1]+data_range[0])/2.0
    return t, y

def fit_ellipse(x, y, method='RANSAC', w=None):
    raise NotImplementedError   # better to use a method in the OpenCV
    if w is None:
        w = []
    if method=='least_square':
        A = np.concatenate((x**2, x*y, y**2, x, y), axis=1)
        b = np.ones_like(x)

        # Modify A,b for weighted least squares
        if len(w) == len(x):
            W = np.diag(w)
            A = np.dot(W, A)
            b = np.dot(W, b)

        # Solve by method of least squares
        c = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

        # Get circle parameters from solution
        A0 = c[0]
        B0 = c[1] / 2
        C0 = c[2]
        D0 = c[3] / 2
        E0 = c[4] / 2
    elif method=='RANSAC':
        A = np.concatenate((x**2, x*y, y**2, x), axis=1)
        b = -2*y
        ransac = linear_model.RANSACRegressor()
        ransac.fit(A, b)
        c0, c1, c2, c3 = ransac.estimator_.coef_[0]
        c4 = ransac.estimator_.intercept_[0]
        E0 = -1/c4
        A0 = c0*E0
        B0 = c1*E0/2
        C0 = c2*E0
        D0 = c3*E0/2
    else:
        raise ValueError

    # center of ellipse
    cx = (C0*D0 - B0*E0)/(B0**2 - A0*C0)
    cy = (A0*E0 - B0*D0)/(B0**2 - A0*C0)
    temp = 1.0 - A0*cx**2 - 2.0*B0*cx*cy - C0*cy**2 - 2.0*D0*cx - 2.0*E0*cy
    A1 = A0/temp
    B1 = B0/temp
    C1 = C0/temp

    # rotating angle of ellipse
    M = A1**2 + C1**2 + 4*B1**2 - 2*A1*C1
    theta = np.arcsin(np.sqrt((-(C1-A1)*np.sqrt(M) + M)/(2*M)))

    # length of axis of ellipse
    a = np.sqrt(1.0/(A1*np.cos(theta)**2 + 2*B1*np.cos(theta)*np.sin(theta)+C1*np.sin(theta)**2))
    b = np.sqrt(1.0/(A1*np.sin(theta)**2 - 2*B1*np.sin(theta)*np.cos(theta)+C1*np.cos(theta)**2))
    return cx,cy, a,b, theta


if __name__ == '__main__':
    # calculate_transformation()
    # filename = '/home/hwangmh/pycharmprojects/FLSpegtransfer/vision/coordinate_pairs.npy'
    # data = np.load(filename)
    # print(data)

    pts1 = [[0, 1, 0], [1, 0, 0], [0, -1, 0]]
    pts2 = [[-0.7071, 0.7071, 0], [0.7071, 0.7071, 0], [0.7071, -0.7071, 0]]
    T = get_rigid_transform(pts1, pts2)
    print(T)

    # f = 6  # (Hz)
    # A = 1  # amplitude
    # t, waveform = create_waveform(interp=[0.1, 0.5], amp1=A, amp2=A * 3, amp3=A * 4, freq1=f, freq2=f * 1.8,
    #                               freq3=f * 1.4, phase=0.0, step=200)
    # t, waveform = create_waveform(interp=[0.1, 0.5], amp1=A, amp2=A * 1.2, amp3=A * 4.2, freq1=0.8 * f, freq2=f * 1.9,
    #                               freq3=f * 1.2, phase=0.5, step=200)
    # t, waveform = create_waveform(interp=[0.1, 0.5], amp1=A, amp2=A * 1.5, amp3=A * 3.5, freq1=f, freq2=f * 1.8,
    #                               freq3=f * 1.3, phase=0.3, step=200)