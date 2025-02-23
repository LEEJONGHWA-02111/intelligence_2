#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO

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

def main():
    # --------- 조정 가능한 파라미터 (실행 중 화면 오버레이와 콘솔 출력으로 확인) ---------
    alpha = 0.4            # smoothing factor: 0.0~1.0 (낮을수록 부드럽게 반응, 값은 실험으로 조정)
    height_standard = 200  # 객체와의 거리를 추정하기 위한 기준값: bounding box 높이가 이 값 미만이면 전진으로 판단
    center_threshold = 20  # 이미지 중심과 객체 중심 사이의 허용 오차 (픽셀 단위)
    angular_gain = 0.005   # 오차에 따른 회전 명령 비례 계수
    linear_speed = 0.1     # 전진 명령에 사용할 선형 속도
    # -----------------------------------------------------------------------------------------

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # YOLO 모델 로드: 실제 엔진 파일 경로로 수정하세요.
    model_path = '/home/amr1_best.engine'  # 예시 경로 – 실제 파일 경로로 변경 필요
    model = YOLO(model_path, task="detect")

    smoothed_box = None
    target_id = None  # 추적할 객체의 tracking id; 초기에는 가장 큰 객체를 선택

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # YOLO를 통해 객체 추적 (ByteTrack 알고리즘 사용)
        results = model.track(source=frame, show=False, tracker='bytetrack.yaml')

        target_box = None
        max_area = 0

        # 결과에서 객체 정보를 추출
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

                # 아직 타겟이 없으면 가장 큰 객체 선택
                if target_id is None:
                    if area > max_area:
                        max_area = area
                        target_box = current_box
                        target_id = current_id
                else:
                    # 이미 타겟이 지정되어 있다면, 해당 tracking id만 선택
                    if current_id == target_id:
                        target_box = current_box
                        break

        if target_box is None:
            print("타겟 객체를 찾지 못했습니다. 타겟을 재설정합니다.")
            target_id = None
            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Bounding box에 대한 temporal smoothing 적용
        smoothed_box = smooth_box(smoothed_box, target_box, alpha)
        cx, cy, w, h = smoothed_box
        image_center_x = frame.shape[1] / 2
        error_x = image_center_x - cx

        # 제어 명령 계산 (실제 로봇 제어 대신 콘솔 출력)
        twist_angular = 0.0
        twist_linear = 0.0

        if abs(error_x) > center_threshold:
            twist_angular = angular_gain * error_x
            twist_linear = 0.0  # 회전 중에는 전진하지 않음
        else:
            if h < height_standard:
                twist_linear = linear_speed
            else:
                twist_linear = 0.0

        print(f"Twist 명령 -> 선형: {twist_linear:.3f}, 각속도: {twist_angular:.3f}")

        # bounding box 그리기
        x1_disp = int(cx - w / 2)
        y1_disp = int(cy - h / 2)
        x2_disp = int(cx + w / 2)
        y2_disp = int(cy + h / 2)
        cv2.rectangle(frame, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 255, 0), 2)

        # 파라미터 디버깅용 텍스트 오버레이
        overlay_text1 = f"BB height: {h:.1f} (std: {height_standard})"
        overlay_text2 = f"Center error: {error_x:.1f} (thresh: {center_threshold})"
        overlay_text3 = f"alpha: {alpha:.2f}"
        cv2.putText(frame, overlay_text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, overlay_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, overlay_text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        label = f"ID: {target_id}, err_x: {error_x:.1f}"
        cv2.putText(frame, label, (x1_disp, y1_disp - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 영상 디스플레이
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
