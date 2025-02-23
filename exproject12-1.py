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

def get_camera_index(max_cam_index=10):
    """
    사용 가능한 카메라 인덱스를 검색합니다.
    0을 제외한 인덱스가 있으면 첫 번째를 반환하고,
    없으면 사용 가능한 첫 번째 인덱스를, 그래도 없으면 0을 반환합니다.
    """
    available_cams = []
    for index in range(max_cam_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cams.append(index)
            cap.release()
    for cam in available_cams:
        if cam != 0:
            return cam
    if available_cams:
        return available_cams[0]
    return 0

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

        # YOLO 모델 로드 – 여기서는 .pt 파일을 사용하지만, 필요 시 engine 파일로 수정하세요.
        self.model = YOLO('/home/seungrok/amr1_best.pt', task="detect")

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
        self.target_id = None     # 추적할 객체의 tracking id
        self.processed_frame = None  # 디버깅용 처리된 이미지

        # 자동으로 사용할 카메라 인덱스 선택
        cam_index = get_camera_index()
        self.get_logger().info(f"Using camera index: {cam_index}")
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            rclpy.shutdown()

        # 로컬 창("Tracked Image") 생성 (한 번만 호출하면 이후 자동 관리)
        cv2.namedWindow("Tracked Image", cv2.WINDOW_NORMAL)
        cv2.startWindowThread()

        # 프레임 처리 스레드 시작 (별도 스레드에서 YOLO 추론)
        threading.Thread(target=self.process_frames, daemon=True).start()
        # Twist 명령 발행 타이머 (예: 10Hz)
        self.twist_timer = self.create_timer(0.1, self.publish_twist)
        # 이미지 발행 타이머 (예: 10Hz)
        self.image_timer = self.create_timer(0.1, self.publish_tracked_image)

    def process_frames(self):
        """웹캠 프레임을 읽어 YOLO 추론 수행 후, 객체 정보를 업데이트하는 스레드"""
        smoothed_box = None
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to read frame")
                time.sleep(0.1)  # 읽기 실패 시 대기 시간을 늘림
                continue

            # YOLO 추적 수행 (ByteTrack tracker 사용)
            results = self.model.track(source=frame, show=False, tracker='bytetrack.yaml')
            target_box = None
            max_area = 0

            # 결과에서 객체 정보 추출 (가장 큰 객체 또는 기존 target_id와 일치하는 객체)
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
                        if area > max_area:
                            max_area = area
                            target_box = current_box
                            self.target_id = current_id
                    else:
                        if current_id == self.target_id:
                            target_box = current_box
                            break

            # 만약 타겟 객체를 찾지 못하면 target_id 초기화 후 반복
            if target_box is None:
                with self.lock:
                    self.target_id = None
                time.sleep(0.1)  # 타겟을 찾지 못했을 때 대기 시간 늘림
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

            # 공유 변수 업데이트 (Lock 사용)
            with self.lock:
                self.latest_box = smoothed_box
                self.latest_error_x = error_x
                self.latest_height = height
                self.processed_frame = annotated_frame

            time.sleep(0.1)  # YOLO 인식 주기를 조절하기 위한 추가 대기

    def publish_twist(self):
        """타이머 콜백: 최신 정보를 이용해 Twist 명령을 계산하고 발행"""
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
        else:  # height > self.desired_max_height
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

        # CompressedImage 메시지 생성 (JPEG 압축)
        msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
        self.image_pub.publish(msg)

        # 로컬 창에 표시
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
