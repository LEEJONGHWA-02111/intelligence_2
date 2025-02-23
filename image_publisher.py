# ROS 2를 활용하여 카메라에서 영상을 읽고, 이를 ROS Image 메시지로 퍼블리싱(Publishing)하는 노드
# OpenCV를 사용하여 카메라에서 영상을 가져오고, 10Hz의 주기로 camera_image 토픽으로 이미지를 전송

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera_image', 10)
        timer_period = 0.1  # Publish at 10 Hz
        self.timer = self.create_timer(timer_period, self.publish_image)
        self.cap = cv2.VideoCapture('/dev/video0')  # Use the default camera (change the index if needed)
        self.bridge = CvBridge()

        if not self.cap.isOpened():
            self.get_logger().error('Failed to open camera')
            raise RuntimeError('Camera not accessible')

    def publish_image(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert OpenCV image (BGR) to ROS Image message
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
            self.get_logger().info('Published image')
        else:
            self.get_logger().warn('Failed to capture image')

    def destroy_node(self):
        self.cap.release()  # Release the camera when shutting down
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()

    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
