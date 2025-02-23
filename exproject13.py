#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

def smooth_box(prev_box, current_box, alpha):
    """
    Exponential smoothing을 적용하여 bounding box 값을 보정합니다.
    prev_box: 이전 프레임의 [center_x, center_y, width, height] (None일 수 있음)
    current_box: 현재 프레임의 [center_x, center_y, width, height]
    alpha: smoothing factor (0.0 ~ 1.0; 낮을수록 더 부드럽게 반응)
    """
    if prev_box is None:
        return current_box
    return alpha * np.array(current_box) + (1 - alpha) * np.array(prev_box)

class ImprovedObjectFollower(Node):
    def __init__(self):
        super().__init__('improved_object_follower')

        # Publisher: Twist 명령 (/cmd_vel) – 실제 로봇 제어 시 사용
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Publisher: 처리된 이미지 (/tracked_image) – 압축된 이미지 (CompressedImage)
        self.image_pub = self.create_publisher(CompressedImage, '/tracked_image', 10)

        self.bridge = CvBridge()

        # YOLO 모델 로드 – 여기서는 학습된 pt 파일을 사용 (필요 시 engine 파일로 수정)
        self.model = YOLO('/home/rokey3/amr1_best.engine', task="detect")

        # ---------------- 파라미터 (실험으로 조정) ----------------
        self.alpha = 0.4                     # smoothing factor (0.0 ~ 1.0)
        self.desired_min_height = 240        # bounding box 높이가 240 미만이면 객체가 멀다고 판단 → 전진
        self.desired_max_height = 270        # bounding box 높이가 270 초과하면 객체가 너무 가까워 → 후진
        self.center_threshold = 20           # 데드밴드: 이미지 중심과 객체 중심의 허용 오차 (픽셀 단위)
        self.center_threshold_mid = 50       # 중간 오차 범위 경계
        self.angular_gain_low = 0.003        # 20~50픽셀 구간에서 적용할 각속도 제어 계수 (느린 회전)
        self.angular_gain_high = 0.005       # 50픽셀 초과 시 적용할 각속도 제어 계수 (빠른 회전)
        self.linear_speed = 0.05             # 선형 속도 (전진/후진)
        # -----------------------------------------------------------

        # 공유 변수 (Lock으로 보호)
        self.lock = threading.Lock()
        self.latest_box = None    # [center_x, center_y, width, height]
        self.latest_error_x = 0.0
        self.latest_height = 0.0
        self.target_id = None     # 추적할 객체의 tracking id (최초 한 번만 설정)
        self.processed_frame = None  # 디버깅용 처리된 이미지
        self.latest_frame = None     # 구독한 카메라 원본 프레임

        # 카메라 토픽 구독 (터틀봇 내 카메라가 발행하는 토픽)
        self.create_subscription(CompressedImage, '/camera/image/compressed', self.image_callback, 10)

        # 디버깅용 로컬 창 생성
        cv2.namedWindow("Tracked Image", cv2.WINDOW_NORMAL)
        cv2.startWindowThread()

        # 프레임 처리 스레드 시작 (별도 스레드에서 YOLO 추론)
        threading.Thread(target=self.process_frames, daemon=True).start()
        # Twist 명령 발행 타이머 (예: 10Hz)
        self.twist_timer = self.create_timer(0.1, self.publish_twist)
        # 이미지 발행 타이머 (예: 10Hz)
        self.image_timer = self.create_timer(0.1, self.publish_tracked_image)

    def image_callback(self, msg):
        """카메라 토픽으로부터 받은 CompressedImage 메시지를 cv2 이미지로 변환하여 저장"""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_frame = cv_image
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def process_frames(self):
        """
        구독한 카메라 프레임을 사용하여 YOLO 추론을 수행하고,
        타겟 객체의 bounding box 정보를 업데이트하는 스레드
        """
        smoothed_box = None
        while rclpy.ok():
            with self.lock:
                if self.latest_frame is None:
                    time.sleep(0.1)
                    continue
                frame = self.latest_frame.copy()

            # YOLO 추적 수행 (ByteTrack tracker 사용)
            results = self.model.track(source=frame, show=False, tracker='bytetrack.yaml')
            target_box = None

            # 타겟 ID가 아직 설정되지 않았다면, 한 번만 for 루프를 돌려 가장 큰 객체로 설정
            if self.target_id is None:
                max_area = 0
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
                        if area > max_area:
                            max_area = area
                            target_box = current_box
                            self.target_id = current_id  # 최초 한 번만 target_id 설정
            else:
                # target_id가 설정되어 있다면 해당 id에 해당하는 객체만 검색
                found = False
                for result in results:
                    for box in result.boxes:
                        current_id = int(box.id[0]) if box.id is not None else None
                        if current_id == self.target_id:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1
                            target_box = [center_x, center_y, width, height]
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # 만약 target_id에 해당하는 객체가 보이지 않으면 target_id를 초기화하여 다음 프레임부터 다시 설정
                    with self.lock:
                        self.target_id = None
                    time.sleep(0.1)
                    continue

            # Bounding box smoothing 적용
            if smoothed_box is None:
                smoothed_box = target_box
            else:
                smoothed_box = smooth_box(smoothed_box, target_box, self.alpha)

            image_center_x = frame.shape[1] / 2
            error_x = image_center_x - smoothed_box[0]
            height = smoothed_box[3]

            # 처리된 이미지: YOLO가 그린 annotated image (없으면 원본 프레임)
            annotated_frame = results[0].plot() if results and results[0].boxes is not None else frame

            with self.lock:
                self.latest_box = smoothed_box
                self.latest_error_x = error_x
                self.latest_height = height
                self.processed_frame = annotated_frame

            time.sleep(0.1)  # 인식 주기 조절을 위한 대기

    def publish_twist(self):
        """타이머 콜백: 최신 정보를 이용해 Twist 명령을 계산 후 발행"""
        with self.lock:
            if self.latest_box is None:
                return
            error_x = self.latest_error_x
            height = self.latest_height

        twist = Twist()
        # 각속도 제어: 바운딩 박스 중심 오차에 따라 회전 속도 조절
        abs_error = abs(error_x)
        if abs_error <= self.center_threshold:
            twist.angular.z = 0.0
        elif abs_error <= self.center_threshold_mid:
            twist.angular.z = self.angular_gain_low * error_x  # 중간 오차: 느린 회전
        else:
            twist.angular.z = self.angular_gain_high * error_x  # 큰 오차: 빠른 회전

        # 선형 속도 제어: bounding box 높이에 따른 세 구간 제어
        if height < self.desired_min_height:
            twist.linear.x = self.linear_speed      # 객체가 멀면 전진
        elif self.desired_min_height <= height <= self.desired_max_height:
            twist.linear.x = 0.0                      # 적정 거리면 정지
        else:
            twist.linear.x = -self.linear_speed       # 객체가 너무 가까우면 후진

        self.cmd_vel_pub.publish(twist)
        self.get_logger().debug(f"Twist: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f}")

    def publish_tracked_image(self):
        """타이머 콜백: 처리된 이미지를 /tracked_image 토픽(CompressedImage)으로 발행하고, 로컬 창에 표시"""
        with self.lock:
            if self.processed_frame is None or self.latest_box is None:
                return
            frame = self.processed_frame.copy()
            box = self.latest_box
            error_x = self.latest_error_x
            height = self.latest_height

        cx, cy, w, h_box = box
        x1 = int(cx - w/2)
        y1 = int(cy - h_box/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h_box/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        overlay1 = f"BB height: {height:.1f} (desired: {self.desired_min_height}-{self.desired_max_height})"
        overlay2 = f"Center error: {error_x:.1f} (thresh: {self.center_threshold})"
        overlay3 = f"alpha: {self.alpha:.2f}"
        cv2.putText(frame, overlay1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, overlay2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, overlay3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
        self.image_pub.publish(msg)

        cv2.imshow("Tracked Image", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImprovedObjectFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
