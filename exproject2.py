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
        # cmd_vel publisher: 로봇의 선형 및 각속도 명령 전송
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # 카메라 이미지 구독 (AMR_camera)
        self.image_sub = self.create_subscription(CompressedImage, 'AMR_image', self.image_callback, 10)
        self.bridge = CvBridge()
        # YOLO 모델 로드 (모델 경로는 실제 모델 경로로 수정)
        self.model = YOLO('amr1_best.engine')
        
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

        # 타겟 객체 ID (tracking id) 저장: 시작 시 가장 큰 객체를 선택하여 저장함.
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
        # 이미지 디코딩
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        # YOLO tracking: 결과로 각 객체의 bounding box 및 tracking id를 포함하는 결과 반환
        results = self.model.track(source=frame, show=False, tracker='bytetrack.yaml')
        
        target_box = None
        max_area = 0

        # 각 프레임마다 추출된 모든 객체에 대해 반복
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # tracking id: 반드시 tracking id를 이용하여 객체 식별 (단, box.id가 None이 아닐 때)
                if box.id is not None:
                    current_id = int(box.id[0])
                else:
                    current_id = None

                # bounding box 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                current_box = [center_x, center_y, width, height]

                # 초기 타겟 선택: 아직 target_id가 지정되지 않은 경우 가장 큰 객체 선택
                if self.target_id is None:
                    if area > max_area:
                        max_area = area
                        target_box = current_box
                        self.target_id = current_id  # 타겟 객체의 tracking id 저장
                else:
                    # 이미 타겟 객체가 지정된 경우, 해당 tracking id와 일치하는 객체만 사용
                    if current_id == self.target_id:
                        target_box = current_box
                        break  # 타겟을 찾으면 더 이상 다른 객체는 볼 필요 없음
        
        # 타겟 객체가 없으면 타겟 잃었다고 로그 남기고, target_id 초기화하여 재탐색
        if target_box is None:
            self.get_logger().info("타겟 객체를 찾지 못했습니다. 다시 탐색합니다.")
            self.target_id = None
            return

        # Temporal smoothing 적용
        filtered_box = self.smooth_box(target_box)

        # 이미지의 가로 중심 좌표
        image_center_x = frame.shape[1] / 2
        # 객체 중심과 이미지 중심 정렬 및 전진 제어 수행
        self.control_robot(filtered_box, image_center_x)

    def control_robot(self, box, image_center_x):
        """
        객체의 bounding box 정보를 기반으로 로봇의 이동(회전, 전진) 제어
        box: [center_x, center_y, width, height]
        image_center_x: 이미지의 가로 중앙 좌표
        """
        center_x, center_y, width, height = box
        twist = Twist()
        
        # 1. 카메라 중앙과 객체 중심의 오차 계산 (좌우 오차)
        error_x = image_center_x - center_x  # 양수면 객체가 왼쪽, 음수면 오른쪽
        
        # 만약 오차가 허용 범위 내에 있지 않으면 회전만 수행
        if abs(error_x) > self.center_threshold:
            twist.angular.z = self.angular_gain * error_x
            twist.linear.x = 0.0  # 회전 시 전진 정지
            self.get_logger().debug(f"회전 명령: angular.z={twist.angular.z:.3f}, error_x={error_x:.1f}")
        else:
            # 2. 오차가 허용 범위 내에 있을 경우, bounding box 높이로 전진/정지 결정
            if height < self.height_standard:
                twist.linear.x = self.linear_speed  # 객체가 멀면 전진
                twist.angular.z = 0.0
                self.get_logger().debug(f"전진 명령: linear.x={twist.linear.x:.2f}, height={height:.1f}")
            else:
                twist.linear.x = 0.0  # 객체가 가까워지면 정지
                twist.angular.z = 0.0
                self.get_logger().debug(f"정지 명령: height={height:.1f} >= height_standard ({self.height_standard})")
        
        # cmd_vel 토픽에 Twist 메시지 발행
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
