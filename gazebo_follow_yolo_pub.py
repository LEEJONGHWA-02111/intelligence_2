import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import time
import threading
import math
from geometry_msgs.msg import Twist

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/robot/camera/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        
        self.publisher_ = self.create_publisher(Image, 'processed_image', 10)
        
        self.model = YOLO('amr_best.pt')
        self.coordinates = []
        
        self.current_frame = None
        self.frame_count = 0
        self.processed_frame = None
        self.classNames = ['car']
        
        self.lock = threading.Lock()
        threading.Thread(target=self.process_frames, daemon=True).start()
        
        self.timer = self.create_timer(0.1, self.publish_image)
        
        
        
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot/cmd_vel', 10)
        self.timer1 = self.create_timer(0.1, self.publish_twist)
        self.center_threshold = 10 
        self.angular_gain = 0.005
        self.height_standard = 200
        self.linear_speed = 0.1 
        # 예시 클래스 내에서 추적할 track_id를 설정
        self.target_track_id = None  # 추적할 track_id를 None으로 초기화
        self.boxes = []  

        
    def publish_twist(self):
        
        twist = Twist()
        with self.lock:
            if len(self.boxes) > 0:
                # 기존 처리 로직
                # 예: 중심 추적, 로봇 제어 등
                error_x = self.current_frame.shape[1] / 2 - self.x  
                
                if abs(error_x) > 10:
                    twist.angular.z = 0.05 * error_x
                else:
                    twist.angular.z = 0.0
                    
                
                # 오차가 허용 범위를 벗어나면 회전 명령 실행
                if abs(error_x) > self.center_threshold:
                    twist.angular.z = self.angular_gain * error_x
                
                if (self.height_standard -50) < self.h < self.height_standard:
                    twist.linear.x = 0.0
                    
                elif self.h >= self.height_standard:
                    twist.linear.x = (-1) * self.linear_speed
                    
                elif self.h <= (self.height_standard -50):
                    twist.linear.x = (1) * self.linear_speed
                    
                self.cmd_vel_pub.publish(twist)
            else:
                # box가 비었을 때의 처리 로직 (필요에 따라 추가)
                pass
            



    def process_frames(self):
        """Process frames for object detection in a separate thread."""

        while True:

            if self.current_frame is not None and self.frame_count % 2 == 0:
                with self.lock:
                    frame = self.current_frame.copy()
                       
                start_time = time.time() 
                results = self.model.track(frame, persist=True)
                end_time = time.time()  
                
                if results[0].boxes is None:
                    continue 
                
                inference_time = end_time - start_time
                print(f"Inference Time: {inference_time:.4f} seconds")
                
                
                self.boxes = results[0].boxes.xywh.cpu()
                boxes = results[0].boxes.xywh.cpu()
                
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    track_ids = []
                
                
                annotated_frame = results[0].plot()
                
                    
                if self.target_track_id not in track_ids:
                    self.target_track_id = None  
                    
                if boxes.numel() > 0:
                    for box, track_id in zip(boxes, track_ids):
                        if self.target_track_id is None:
                            self.target_track_id = track_id
                            self.x, self.y, self.w, self.h = box.tolist()
                        elif track_id == self.target_track_id:
                            self.x, self.y, self.w, self.h = box.tolist()

                

                with self.lock:
                    self.processed_frame = annotated_frame

            self.frame_count += 1
            time.sleep(0.05)
                    
                    
    def publish_image(self):
        """Publish the latest processed frame."""
        if self.processed_frame is not None:
            with self.lock:
                ros_image = self.bridge.cv2_to_imgmsg(self.processed_frame, encoding="bgr8")
            self.publisher_.publish(ros_image)
        
    def image_callback(self, msg):
        try:
            # ROS Image 메시지를 OpenCV 형식으로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # 이미지 출력
            
            self.current_frame = cv_image.copy()
            # cv2.imshow("Camera Image", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
