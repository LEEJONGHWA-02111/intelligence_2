import cv2
import numpy as np
from ultralytics import YOLO

def smooth_box(previous_box, current_box, alpha=0.4):
    """
    Exponential smoothing을 적용하여 bounding box 값을 보정합니다.
    previous_box: 이전에 저장된 [center_x, center_y, width, height] (또는 None)
    current_box: 현재 프레임에서 얻은 [center_x, center_y, width, height]
    alpha: smoothing factor (0.0 ~ 1.0; 낮을수록 더 부드럽게 반응)
    """
    if previous_box is None:
        return current_box
    return alpha * np.array(current_box) + (1 - alpha) * np.array(previous_box)

def main():
    # ----------- 조정 가능한 파라미터 (필요 시 수정) -----------
    alpha = 0.4              # smoothing factor: 0.0~1.0, 낮을수록 더 부드럽게 반응
    height_standard = 200    # bounding box 높이가 이 값 미만이면 전진, 이 값 이상이면 정지
    angular_gain = 0.005     # 좌우 오차에 따른 회전 게인
    linear_speed = 0.1       # 전진 속도
    center_threshold = 10    # 이미지 중심과 객체 중심의 허용 오차 (픽셀 단위)
    # ------------------------------------------------------------

    # YOLO 모델 로드 (TensorRT engine 사용; task 인자 "detect"를 명시)
    model = YOLO('/path/to/amr_best.engine', task="detect")
    
    # 카메라(웹캠) 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    smoothed_box = None
    target_id = None  # 타겟 객체의 tracking id (초기에는 미지정)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break
        
        # YOLO 모델을 사용하여 추적 수행 (ByteTrack 사용)
        results = model.track(source=frame, show=False, tracker='bytetrack.yaml')
        
        target_box = None
        max_area = 0

        # 프레임 내 모든 객체 탐색
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # tracking id를 사용하여 객체 식별 (없으면 None)
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
                
                # 타겟 객체가 아직 미지정된 경우, 가장 큰 객체를 선택
                if target_id is None:
                    if area > max_area:
                        max_area = area
                        target_box = current_box
                        target_id = current_id
                else:
                    # 이미 타겟이 지정되어 있다면 해당 tracking id와 일치하는 객체만 선택
                    if current_id == target_id:
                        target_box = current_box
                        break

        if target_box is None:
            print("타겟 객체를 찾지 못했습니다. 타겟을 재설정합니다.")
            target_id = None
        else:
            # Temporal smoothing 적용
            smoothed_box = smooth_box(smoothed_box, target_box, alpha)
            # 이미지 가로 중심 계산
            image_center_x = frame.shape[1] / 2
            center_x, center_y, width, height = smoothed_box

            # 이미지 중심과 객체 중심의 오차 계산
            error_x = image_center_x - center_x

            # 제어 명령 계산 (실제 로봇에서는 이 값을 cmd_vel에 publish)
            twist_angular = 0.0
            twist_linear = 0.0
            if abs(error_x) > center_threshold:
                twist_angular = angular_gain * error_x
            else:
                if height < height_standard:
                    twist_linear = linear_speed
                else:
                    twist_linear = 0.0

            # 시뮬레이션: 계산된 명령을 콘솔에 출력
            print(f"Twist 명령 -> 선형: {twist_linear:.3f}, 각속도: {twist_angular:.3f}")

            # 화면에 bounding box와 정보 표시
            x1_disp = int(center_x - width / 2)
            y1_disp = int(center_y - height / 2)
            x2_disp = int(center_x + width / 2)
            y2_disp = int(center_y + height / 2)
            cv2.rectangle(frame, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 255, 0), 2)
            cv2.putText(frame, f"Target: ID {target_id}", (x1_disp, y1_disp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"오차: {error_x:.1f}", (x1_disp, y1_disp - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 카메라 피드 디스플레이
        cv2.imshow("Camera Feed", frame)
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
