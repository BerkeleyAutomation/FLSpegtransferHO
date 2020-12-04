from FLSpegtransfer.vision.ZividCapture import ZividCapture
import cv2
import numpy as np

zc_overhead = ZividCapture(which_camera="overhead")
zc_inclined = ZividCapture(which_camera="inclined")
zc_overhead.start()
zc_inclined.start()

# while True:
#     image = zc.capture_2Dimage(color='BGR')
#     cv2.imshow("", image)
#     cv2.waitKey(1)

# check images
img_color1, img_depth1, img_point1 = zc_overhead.capture_3Dimage(color='BGR')
img_color2, img_depth2, img_point2 = zc_inclined.capture_3Dimage(color='BGR')
# zc_overhead.display_rgb(img_color1, block=False)
# zc_overhead.display_rgb(img_color2, block=True)
# zc.display_depthmap(img_point)
# zc.display_pointcloud(img_point, img_color)

# cv2.imwrite("color_overhead.png", img_color1)
# np.save("img_color_overhead", img_color1)
# np.save("img_depth_overhead", img_depth1)
# np.save("img_point_overhead", img_point1)
cv2.imwrite("color_inclined.png", img_color2)
np.save("img_color_inclined", img_color2)
np.save("img_depth_inclined", img_depth2)
np.save("img_point_inclined", img_point2)
