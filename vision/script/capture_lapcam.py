import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


cap = cv2.VideoCapture(0)
cnt = 0
while True:
    ret, img = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('\r'):  # ENTER
        cv2.imwrite("img"+str(int(cnt))+".png", img)
        print("img"+str(int(cnt))+".png saved")
        cnt += 1
    elif key == ord('q'):  # ESD
        break

    cv2.imshow("Image", img)
    cv2.waitKey(1)