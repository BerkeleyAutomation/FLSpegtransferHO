import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ImagePublisher:
    def __init__(self):
        self.img_left_pub = rospy.Publisher("/img_left", Image, latch=True, queue_size=1)
        self.img_right_pub = rospy.Publisher("/img_right", Image, latch=True, queue_size=1)

        self.bridge = CvBridge()
        self.img_left = cv2.imread('img_left0.png')
        self.img_right = cv2.imread('img_right0.png')

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('img_pub_node', anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

        self.main()

    def img_publish(self):
        try:
            self.img_left_pub.publish(self.bridge.cv2_to_imgmsg(self.img_left, "rgb8"))
            self.img_right_pub.publish(self.bridge.cv2_to_imgmsg(self.img_right, "rgb8"))
        except CvBridgeError as e:
            print(e)

    def main(self):
        while True:
            self.img_publish()

if __name__ == '__main__':
    pub = ImagePublisher()
