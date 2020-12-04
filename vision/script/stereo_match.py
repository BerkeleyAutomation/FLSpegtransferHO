#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
import numpy as np
import cv2 as cv

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    print('loading images...')
    # imgL = cv.pyrDown(cv.imread('/home/maggie/opencv-master/samples/data/img0_left.jpg'))  # downscale images for faster processing
    # imgR = cv.pyrDown(cv.imread('/home/maggie/opencv-master/samples/data/img0_right.jpg'))
    imgL = cv.imread("img_left_rect.pgm")
    imgR = cv.imread("img_right_rect.pgm")

    w = imgL.shape[1]
    h = imgL.shape[0]
    print(w, h)

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 2144, #8*3*window_size**2,
        P2 = 1117, #32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 0 #100, #set to 0 to disable speckle filtering
        #speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # print('generating 3d point cloud...',)
    # h, w = imgL.shape[:2]
    # f = 0.8*w                          # guess for focal length
    # #Q = np.float32([[1, 0, 0, -0.5*w],
    # #                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
    # #                [0, 0, 0,     -f], # so that y-axis looks up
    # #                [0, 0, 1,      0]])
    # cx = 828.79
    # cy = 581.30
    # cx2 = 1048.27
    # cy2 = 594.88
    # f = 1720.00
    # Tx = 0.48816596
    # # Tx = 5.85799151
    # Q = np.float32([[1, 0, 0, -cx], #1, 0, 0, -cx
    #               [0, 1, 0, -cy], #0, 1, 0, -cy
    #               [0, 0, 0, f], #0, 0, 0, f
    #               [0, 0, -1./Tx, (cx-cx2)/Tx]]) #0, 0, -1/Tx, (cx-cx2)/Tx
    # points = cv.reprojectImageTo3D(disp, Q)
    # colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    # mask = disp > disp.min()
    # out_points = points[mask]
    # out_colors = colors[mask]
    # out_fn = 'out.ply'
    # write_ply(out_fn, out_points, out_colors)
    # print('%s saved' % out_fn)

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
