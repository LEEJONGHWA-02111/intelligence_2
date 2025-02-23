import warnings
# Axes3D 경고를 무시 (3D plot 기능을 사용하지 않는다면)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.projections")

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class ObjectFollower(Node):
    def __init__(self):
        super().__init__('object_follower')
        # cmd_vel 토픽으로 로봇의 선형 및 각속도 명령 발행
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # 카메라 이미지 구독 (AMR_camera)
        self.image_sub = self.create_subscription(CompressedImage, 'AMR_image', self.image_callback, 10)
        self.bridge = CvBridge()
        # YOLO 모델 로드 (TensorRT engine 사용: 경로 및 task 인자 수정)
        self.model = YOLO('amr1_best.engine', task="detect")
        
        # ---------- Temporal Smoothing 관련 파라미터 ----------
        self.alpha = 0.4  # <<-- smoothing factor: 0.0~1.0, 낮을수록 더 부드럽게 반응 (조정 가능)
        self.smoothed_box = None  # [center_x, center_y, width, height]
        # ---------------------------------------------------------
        
        # ---------- 전진/정지 기준 파라미터 ----------
        self.height_standard = 200  # <<-- bounding box의 높이가 이 값 미만이면 전진, 범위 내이면 정지 (조정 가능)
        # ------------------------------------------------

        # ---------- 제어 게인 (비례 제어 계수) ----------
        self.angular_gain = 0.005  # <<-- 좌우 오차에 따른 회전 게인 (조정 가능)
        self.linear_speed = 0.1    # <<-- 전진 속도 (조정 가능)
        self.center_threshold = 10 # <<-- 이미지 중심과 객체 중심의 오차 허용 범위 (픽셀 단위, 조정 가능)
        # ------------------------------------------------

        # 타겟 객체 ID (tracking id)를 저장 (시작 시 가장 큰 객체 선택)
        self.target_id = None

    def smooth_box(self, current_box):
        """
        Exponential smoothing을 통해 현재 bounding box 값을 보정합니다.
        current_box: [center_x, center_y, width, height]
        """
        if self.smoothed_box is None:
            self.smoothed_box = current_box
        else:
            self.smoothed_box = self.alpha * np.array(current_box) + (1 - self.alpha) * np.array(self.smoothed_box)
        return self.smoothed_box

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        # --- 카메라 화면 표시 (디버깅용) ---
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1)
        # ------------------------------------

        # YOLO를 통해 추적: ByteTrack 알고리즘 사용
        results = self.model.track(source=frame, show=False, tracker='bytetrack.yaml')
        
        target_box = None
        max_area = 0

        # 프레임 내의 모든 객체에 대해 탐색
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # tracking id를 사용해 객체 식별 (box.id가 있을 경우)
                if box.id is not None:
                    current_id = int(box.id[0])
                else:
                    current_id = None

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                current_box = [center_x, center_y, width, height]

                # 타겟 객체 미지정 시 가장 큰 객체 선택
                if self.target_id is None:
                    if area > max_area:
                        max_area = area
                        target_box = current_box
                        self.target_id = current_id  # 타겟 객체의 tracking id 저장
                else:
                    # 이미 타겟이 지정되어 있다면, 해당 tracking id를 가진 객체만 선택
                    if current_id == self.target_id:
                        target_box = current_box
                        break
        
        if target_box is None:
            self.get_logger().info("타겟 객체를 찾지 못했습니다. 타겟 재설정.")
            self.target_id = None
            return

        # Temporal smoothing 적용
        filtered_box = self.smooth_box(target_box)
        # 이미지 가로 중심 좌표 계산
        image_center_x = frame.shape[1] / 2
        # 제어 로직 실행: 회전 및 전진 명령 결정
        self.control_robot(filtered_box, image_center_x)

    def control_robot(self, box, image_center_x):
        """
        객체의 bounding box 정보를 기반으로 로봇의 이동(회전, 전진)을 제어합니다.
        box: [center_x, center_y, width, height]
        image_center_x: 이미지의 가로 중앙 좌표
        """
        center_x, center_y, width, height = box
        twist = Twist()
        
        # 1. 이미지 중심과 객체 중심의 오차 계산 (좌우 오차)
        error_x = image_center_x - center_x  # 양수면 객체가 왼쪽, 음수면 오른쪽
        
        # 오차가 허용 범위를 벗어나면 회전 명령 실행
        if abs(error_x) > self.center_threshold:
            twist.angular.z = self.angular_gain * error_x
            twist.linear.x = 0.0  # 회전 중에는 전진하지 않음
            self.get_logger().debug(f"회전: angular.z = {twist.angular.z:.3f}, error_x = {error_x:.1f}")
        else:
            # 2. 오차가 허용 범위 내면 bounding box 높이를 기준으로 전진/정지 결정
            if height < self.height_standard:
                twist.linear.x = self.linear_speed  # 객체가 멀면 전진
                twist.angular.z = 0.0
                self.get_logger().debug(f"전진: linear.x = {twist.linear.x:.2f}, height = {height:.1f}")
            else:
                twist.linear.x = 0.0  # 객체가 가까워지면 정지
                twist.angular.z = 0.0
                self.get_logger().debug(f"정지: height = {height:.1f} >= height_standard ({self.height_standard})")
        
        # 계산된 Twist 메시지를 cmd_vel 토픽에 발행
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
