# YOLOv8을 사용하여 실시간 객체 탐지(Object Detection) 수행 후, 결과 이미지를 ROS 2 토픽(processed_image)으로 퍼블리싱하는 ROS 2 노드
# 멀티스레딩(threading) 기법을 활용하여 카메라 프레임 캡처와 YOLO 추론을 분리하여 효율성을 극대화

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import threading
import time
import math

class YoloPublisher(Node):
    def __init__(self):
        super().__init__('yolo_publisher')
        self.publisher_ = self.create_publisher(Image, 'processed_image', 10)
        self.bridge = CvBridge()

        self.model = YOLO('/home/rokey/ros3_ws/src/yolo_transfer/yolo_transfer/best.pt')
        self.coordinates = []
        
        # Video capture setup
        self.cap = cv2.VideoCapture('/dev/video0')  # Adjust this to your camera source
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        # Control flags
        self.frame_count = 0  # Used to process every other frame
        self.lock = threading.Lock()
        self.current_frame = None
        self.processed_frame = None
        self.classNames = ['Truck','Dummy']


        # Start threads
        threading.Thread(target=self.capture_frames, daemon=True).start()
        threading.Thread(target=self.process_frames, daemon=True).start()
        self.timer = self.create_timer(0.1, self.publish_image)

    def capture_frames(self):
        """Continuously capture frames from the camera in a separate thread."""
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame.copy()
            time.sleep(0.01)  # Slight delay to control frame rate

    def process_frames(self):
        """Process frames for object detection in a separate thread."""

        color = (255, 0, 0)
        thickness = 2

        while True:

            if self.current_frame is not None and self.frame_count % 2 == 0:
                with self.lock:
                    frame_to_process = self.current_frame.copy()
                    
                # Measure inference time
                start_time = time.time()  # Start timing
                # Run YOLO object detection on every other frame
                results = self.model(frame_to_process, stream=True)
                end_time = time.time()  # End timing

                # Compute elapsed time
                inference_time = end_time - start_time
                print(f"Inference Time: {inference_time:.4f} seconds")

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = math.ceil(box.conf[0] * 100) / 100
                        cls = int(box.cls[0])
                        cv2.rectangle(frame_to_process, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame_to_process, f"{self.classNames[cls]}: {confidence}", (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        
                

                # Update processed frame
                with self.lock:
                    cv2.rectangle(frame_to_process, (100,100), (200,200), color, thickness)
                    self.processed_frame = frame_to_process

            self.frame_count += 1
            time.sleep(0.05)  # Adjust delay if necessary to control processing rate

    def publish_image(self):
        """Publish the latest processed frame."""
        if self.processed_frame is not None:
            with self.lock:
                ros_image = self.bridge.cv2_to_imgmsg(self.processed_frame, encoding="bgr8")
            self.publisher_.publish(ros_image)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
