import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class YOLOTrackingSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_tracking_subscriber')

        # Subscriber 생성 (AMR_image 토픽에서 CompressedImage 구독)
        self.subscription = self.create_subscription(
            CompressedImage,
            'AMR_image',
            self.listener_callback,
            10  # Queue size
        )

        self.bridge = CvBridge()

    def listener_callback(self, msg):
        """퍼블리시된 CompressedImage 메시지를 받아 OpenCV로 디코딩하여 화면에 표시"""
        try:
            # ROS CompressedImage 메시지를 OpenCV 형식으로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 화면에 표시
            cv2.imshow("YOLO Tracking Subscriber", frame)
            cv2.waitKey(1)  # 실시간 영상 갱신 (키 입력 대기)

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLOTrackingSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
