import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import yaml
from geometry_msgs.msg import Point

class YOLOTrackingPublisher(Node):
    def __init__(self):
        super().__init__('yolo_tracking_publisher')
        # 기존 이미지 퍼블리셔 생성
        self.publisher_ = self.create_publisher(CompressedImage, 'AMR_image', 10)
        # 선택한 객체의 좌표를 퍼블리시할 토픽 생성 (Point 메시지)
        self.selected_coord_publisher = self.create_publisher(Point, 'selected_caller_coord', 10)
        self.bridge = CvBridge()
        
        # YOLO 모델 로드
        self.model = YOLO('/home/leejonghwa/Downloads/inteligence2/standing_best.pt')
        # 카메라 인덱스 2번 (필요에 따라 변경)
        self.cap = cv2.VideoCapture(2)
        # 타이머를 이용해 주기적으로 프레임을 처리 (0.1초 간격)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # map.yaml을 로드하여 이미지 좌표와 지도 좌표의 대응점을 얻고 Homography 계산
        with open('/home/leejonghwa/Downloads/inteligence2/map.yaml', 'r') as f:
            map_data = yaml.safe_load(f)
        self.image_points = np.array(map_data['image_points'], dtype=np.float32)
        self.map_points = np.array(map_data['map_points'], dtype=np.float32)
        self.H, status = cv2.findHomography(self.image_points, self.map_points, cv2.RANSAC)

        # GUI 창 생성 및 마우스 이벤트 콜백 등록 (Tracking 창에서 사용자가 객체를 클릭할 수 있음)
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Tracking", self.on_mouse)

        # 현재 프레임의 검출 객체 정보를 저장 (마우스 클릭 시 사용)
        self.current_detections = []

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame from webcam.")
            return

        # YOLO의 track API를 이용해 객체 탐지 및 tracking 수행
        results = self.model.track(source=frame, show=False, tracker='bytetrack.yaml')

        # 매 프레임마다 현재 검출된 객체 정보를 초기화
        self.current_detections = []

        # 결과에서 각 객체 정보를 추출
        for result in results:
            boxes = result.boxes  # 탐지된 bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표 (정수형 변환)
                track_id = int(box.id[0]) if box.id is not None else None  # 추적 ID
                conf = float(box.conf[0])  # 신뢰도
                cls = int(box.cls[0])  # 클래스 ID
                class_name = self.model.names[cls] if cls in self.model.names else "Unknown"

                # 바운딩 박스와 객체 정보를 이미지에 그리기
                color = (0, 255, 0)  # 초록색
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {track_id} {class_name} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 현재 프레임의 검출 정보를 저장 (마우스 이벤트 시, 클릭한 객체 결정에 사용)
                self.current_detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'track_id': track_id,
                    'class_name': class_name,
                    'conf': conf
                })

        # 압축하여 ROS 2 CompressedImage 메시지로 변환 후 퍼블리시
        _, compressed_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        compressed_img_msg = CompressedImage()
        compressed_img_msg.header.stamp = self.get_clock().now().to_msg()
        compressed_img_msg.format = "jpeg"
        compressed_img_msg.data = compressed_frame.tobytes()
        self.publisher_.publish(compressed_img_msg)

        # GUI 창에 결과 영상 출력
        cv2.imshow("Tracking", frame)
        cv2.waitKey(1)

    def on_mouse(self, event, x, y, flags, param):
        # 마우스 왼쪽 버튼 클릭 시, 현재 검출된 객체 중 클릭 위치에 포함되는 객체를 선택
        if event == cv2.EVENT_LBUTTONDOWN:
            for detection in self.current_detections:
                bbox = detection['bbox']
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    # 객체 중심 좌표 계산
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    point = np.array([[[center_x, center_y]]], dtype=np.float32)
                    # Homography를 적용하여 지도상의 좌표로 변환
                    world_point = cv2.perspectiveTransform(point, self.H)
                    world_coord = world_point[0][0]
                    self.get_logger().info(
                        f"Selected object ID: {detection['track_id']}, World Coord: {world_coord}"
                    )

                    # 선택된 객체의 좌표를 Point 메시지로 퍼블리시
                    point_msg = Point()
                    point_msg.x = float(world_coord[0])
                    point_msg.y = float(world_coord[1])
                    point_msg.z = 0.0
                    self.selected_coord_publisher.publish(point_msg)
                    break

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = YOLOTrackingPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
