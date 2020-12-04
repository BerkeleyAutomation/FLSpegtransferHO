import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImageSubscriber:
    def __init__(self):
        self.img_sub = rospy.Subscriber("/img_left", Image, self.img_sub_cb)
        self.bridge = CvBridge()

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('img_sub_node', anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        self.main()

    def img_sub_cb(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)
        print (np.shape(image))
        # cv2.imshow("Image window", image)
        # cv2.waitKey(3)

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    pub = ImageSubscriber()
