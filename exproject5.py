#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class TurtlebotObjectFollower(Node):
    def __init__(self):
        super().__init__('turtlebot_object_follower')
        # 1) 카메라 구독 (Turtlebot3에 맞는 토픽 이름 확인 후 수정)
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',  # 실제 토픽명 확인 필요
            self.image_callback,
            10
        )
        # 2) 추적된 영상을 송신할 퍼블리셔 (PC에서 구독하여 확인)
        self.tracked_image_pub = self.create_publisher(CompressedImage, '/tracked_image', 10)
        # 3) cmd_vel 퍼블리셔 (Turtlebot3 모터 제어)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()

        # YOLO 모델 로드 (TensorRT engine)
        # 실제 경로와 task="detect" 확인
        self.model = YOLO('/home/amr1_best.engine', task="detect")
        
        # ---------- 파라미터(조정 가능) ----------
        self.alpha = 0.4             # smoothing factor
        self.height_standard = 200   # bounding box 높이 기준
        self.angular_gain = 0.005    # 회전 게인
        self.linear_speed = 0.1      # 전진 속도
        self.center_threshold = 20   # 이미지 중심 오차 허용 범위
        # -------------------------------------

        self.smoothed_box = None
        self.target_id = None

    def smooth_box(self, current_box):
        """ Exponential Smoothing """
        if self.smoothed_box is None:
            self.smoothed_box = current_box
        else:
            self.smoothed_box = self.alpha * np.array(current_box) + \
                                (1 - self.alpha) * np.array(self.smoothed_box)
        return self.smoothed_box

    def image_callback(self, msg):
        # 1) 이미지 디코딩
        frame = self.bridge.compressed_imgmsg_to_cv2(msg)

        # 2) YOLO 추적
        results = self.model.track(source=frame, show=False, tracker='bytetrack.yaml')
        
        target_box = None
        max_area = 0
        
        # 3) 가장 큰 객체 or 지정된 id 객체 탐색
        for result in results:
            for box in result.boxes:
                current_id = int(box.id[0]) if box.id is not None else None
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                current_box = [center_x, center_y, width, height]

                if self.target_id is None:
                    # 아직 타겟이 없다면, 가장 큰 객체 선택
                    if area > max_area:
                        max_area = area
                        target_box = current_box
                        self.target_id = current_id
                else:
                    # 타겟이 이미 있다면, 해당 id만 추적
                    if current_id == self.target_id:
                        target_box = current_box
                        break
        
        # 4) 타겟을 찾지 못했다면, id 초기화
        if target_box is None:
            self.get_logger().info("타겟을 찾지 못함, 타겟 재설정")
            self.target_id = None
            # 현재 프레임도 그대로 publish (bounding box 없음)
            self.publish_tracked_image(frame)
            return

        # 5) bounding box smoothing
        filtered_box = self.smooth_box(target_box)
        center_x, center_y, width, height = filtered_box
        image_center_x = frame.shape[1] / 2

        # 6) 제어 명령 계산
        twist = Twist()
        error_x = image_center_x - center_x

        if abs(error_x) > self.center_threshold:
            # 회전
            twist.angular.z = self.angular_gain * error_x
            twist.linear.x = 0.0
        else:
            # 전진/정지
            if height < self.height_standard:
                twist.linear.x = self.linear_speed
                twist.angular.z = 0.0
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
        
        self.cmd_vel_pub.publish(twist)

        # 7) bounding box 그려서 영상 publish
        x1_disp = int(center_x - width / 2)
        y1_disp = int(center_y - height / 2)
        x2_disp = int(center_x + width / 2)
        y2_disp = int(center_y + height / 2)
        cv2.rectangle(frame, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 255, 0), 2)
        label = f"ID: {self.target_id}, err_x={error_x:.1f}"
        cv2.putText(frame, label, (x1_disp, y1_disp - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.publish_tracked_image(frame)

    def publish_tracked_image(self, frame):
        """ bounding box가 그려진 프레임을 /tracked_image 토픽으로 발행 """
        compressed_msg = CompressedImage()
        compressed_msg.header.stamp = self.get_clock().now().to_msg()
        compressed_msg.format = "jpeg"
        _, encoded_frame = cv2.imencode('.jpg', frame)
        compressed_msg.data = encoded_frame.tobytes()
        self.tracked_image_pub.publish(compressed_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotObjectFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
