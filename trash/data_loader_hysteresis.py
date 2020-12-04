import numpy as np

def format_dataset1(q_des, q_act, H_upper, H_lower):
    # Training only using wrist angles
    q_des = q_des[:, 3:]
    q_act = q_act[:, 3:]

    # Angular velocity
    q_des_temp = np.insert(q_des, 0, 0, axis=0)
    q_des_temp = np.delete(q_des_temp, -1, axis=0)
    w_des = q_des - q_des_temp

    # Hysteresis modeling information
    x = []
    set_point = [0.0, 0.0, 0.0]
    track = [0, 0, 0]
    margin_L = [0.0, 0.0, 0.0]
    margin_R = [0.0, 0.0, 0.0]
    for qs, ws in zip(q_des.tolist(), w_des.tolist()):
        set_points = []
        tracks = []
        marginsL = []
        marginsR = []
        for i, q in enumerate(qs):
            if set_point[i] + H_lower[i] < q < set_point[i] + H_upper[i]:
                track[i] = 0
                margin_L[i] = q - (set_point[i] + H_lower[i])
                margin_R[i] = (set_point[i] + H_upper[i]) - q
            elif set_point[i] + H_upper[i] <= q:
                track[i] = 1
                set_point[i] = q - H_upper[i]
                margin_L[i] = H_upper[i] - H_lower[i]
                margin_R[i] = 0.0
            elif q <= set_point[i] + H_lower[i]:
                track[i] = -1
                set_point[i] = q - H_lower[i]
                margin_L[i] = 0.0
                margin_R[i] = H_upper[i] - H_lower[i]
            set_points.append(set_point[i])
            tracks.append(track[i])
            marginsL.append(margin_L[i])
            marginsR.append(margin_R[i])
        x.append(qs + ws + set_points + marginsL + marginsR)
    y = q_act[len(q_act) - len(x):]
    print("x.shape= ", np.shape(x), "y.shape= ", np.shape(y))
    return x, y


def format_dataset2(q_des, q_act, H):
    # Training only using wrist angles
    # q_des = q_des[:, 3:]
    # q_act = q_act[:, 3:]

    # History of joint angles
    # x = [q_(t), q_(t-1), q_(t-2), ..., q_(t-H+1)]
    #     [q_(t+1), q_(t), q_(t-1), ..., q_(t-H+2)]
    #     [ ... ]
    x = []
    for i in range(H):
        x.append(q_des[i:len(q_des)-H+1+i])
    x = x[::-1]
    x = np.hstack(x)
    y = q_act[H-1:]
    print("x.shape= ", np.shape(x), "y.shape= ", np.shape(y))
    return x, y


def format_dataset3(q_des, q_act, H, H_upper, H_lower):
    # Training only using wrist angles
    q_des = q_des[:, 3:]
    q_act = q_act[:, 3:]

    # History of joint angles
    # x = [q_(t), q_(t-1), q_(t-2), ..., q_(t-H+1)]
    #     [q_(t+1), q_(t), q_(t-1), ..., q_(t-H+2)]
    #     [ ... ]
    x = []
    for i in range(H):
        x.append(q_des[i:len(q_des) - H + 1 + i])
    x = x[::-1]
    x = np.hstack(x)

    # Hysteresis modeling information
    x_add = []
    set_point = [0.0, 0.0, 0.0]
    track = [0, 0, 0]
    margin_L = [0.0, 0.0, 0.0]
    margin_R = [0.0, 0.0, 0.0]
    q_des = x[:, :3]
    for qs in q_des.tolist():
        set_points = []
        tracks = []
        marginsL = []
        marginsR = []
        for i, q in enumerate(qs):
            if set_point[i] + H_lower[i] < q < set_point[i] + H_upper[i]:
                track[i] = 0
                margin_L[i] = q - (set_point[i] + H_lower[i])
                margin_R[i] = (set_point[i] + H_upper[i]) - q
            elif set_point[i] + H_upper[i] <= q:
                track[i] = 1
                set_point[i] = q - H_upper[i]
                margin_L[i] = H_upper[i] - H_lower[i]
                margin_R[i] = 0.0
            elif q <= set_point[i] + H_lower[i]:
                track[i] = -1
                set_point[i] = q - H_lower[i]
                margin_L[i] = 0.0
                margin_R[i] = H_upper[i] - H_lower[i]
            set_points.append(set_point[i])
            tracks.append(track[i])
            marginsL.append(margin_L[i])
            marginsR.append(margin_R[i])
        x_add.append(set_points + tracks + marginsL + marginsR)

    x = np.concatenate((x, x_add), axis=1)
    y = q_act[H - 1:]
    print("x.shape= ", np.shape(x), "y.shape= ", np.shape(y))
    return x, y