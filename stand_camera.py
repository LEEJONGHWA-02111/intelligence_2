import cv2
import numpy as np
import torch
import yaml

# ===============================================================
# 1. YOLO 모델 로드 (stand.pt)
# ---------------------------------------------------------------
# ultralytics/yolov5 모델을 torch.hub로 로드하는 예시
# stand.pt는 단일 클래스("caller")만 학습된 모델이라고 가정
model = torch.hub.load('ultralytics/yolov8', 'custom', path='stand_best.pt', force_reload=True)
model.conf = 0.7  # 기본 신뢰도 임계값 설정

# ===============================================================
# 2. map.yaml 파일 로드 및 Homography 행렬 계산
# ---------------------------------------------------------------
# map.yaml에는 이미지와 지도 간 대응점이 미리 정의되어 있어야 함
# 예시 map.yaml 구조:
# image_points:
#   - [x1, y1]
#   - [x2, y2]
#   - [x3, y3]
#   - [x4, y4]
# map_points:
#   - [X1, Y1]
#   - [X2, Y2]
#   - [X3, Y3]
#   - [X4, Y4]

with open('map.yaml', 'r') as f:
    map_data = yaml.safe_load(f)
image_points = np.array(map_data['image_points'], dtype=np.float32)
map_points = np.array(map_data['map_points'], dtype=np.float32)

# Homography 계산: 이미지 좌표를 지도 좌표로 변환하는 행렬 H 구함
H, status = cv2.findHomography(image_points, map_points, cv2.RANSAC)

# ===============================================================
# 3. 카메라 캘리브레이션 파일 로드 (선택 사항)
# ---------------------------------------------------------------
# 만약 'calibration.yaml' 파일이 있다면, 내부 파라미터와 왜곡 계수를 읽어와
# 프레임을 보정하는 데 사용. 파일이 없으면 보정 없이 진행.

try:
    with open('calibration.yaml', 'r') as f:
        calib_data = yaml.safe_load(f)
    camera_matrix = np.array(calib_data['camera_matrix'])
    dist_coeffs = np.array(calib_data['dist_coeffs'])
    print("카메라 캘리브레이션 파일 로드 성공.")
except Exception as e:
    camera_matrix = None
    dist_coeffs = None
    print("캘리브레이션 파일 없음. 보정 없이 진행합니다.")

# ===============================================================
# 4. ByteTrack (또는 BoT-SORT) 객체 추적기 초기화
# ---------------------------------------------------------------
# 이 예시에서는 ByteTrack 라이브러리를 사용한다고 가정.
# 실제 사용 시, 해당 라이브러리를 설치하고 적절한 파라미터를 설정해야 함.
from bytetrack import BYTETracker  # (설치된 라이브러리 사용)
tracker = BYTETracker(track_thresh=0.7)  # tracker 초기화 (추가 파라미터 필요 시 수정)

# 최초 신뢰도 조건을 만족하는 caller를 저장할 변수
first_caller_id = None
first_caller_world_coord = None

# ===============================================================
# 5. 카메라 스트림 시작 및 메인 루프
# ---------------------------------------------------------------
cap = cv2.VideoCapture(0)  # 0번 카메라 (필요에 따라 경로 수정)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    # 5-1. (선택 사항) 캘리브레이션 파라미터가 있다면 프레임 왜곡 보정
    if camera_matrix is not None and dist_coeffs is not None:
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # 5-2. YOLO 모델을 통해 객체 검출
    results = model(frame)
    # 결과는 보통 tensor나 numpy 배열로 제공됨 (예: [x1, y1, x2, y2, conf, class])
    # 여기서는 단일 클래스 "caller"만 있으므로 클래스 정보는 생략해도 됨.
    detections = results.xyxy[0].cpu().numpy()  # 각 row: [x1, y1, x2, y2, conf, class]

    # 5-3. 신뢰도 임계치 필터링 및 tracker 입력 형식 변환
    tracker_inputs = []
    for det in detections:
        conf = det[4]
        if conf >= 0.7:
            x1, y1, x2, y2 = det[:4]
            # ByteTrack에서 요구하는 형식: [x, y, w, h, conf]
            bbox = [x1, y1, x2 - x1, y2 - y1, conf]
            tracker_inputs.append(bbox)
    tracker_inputs = np.array(tracker_inputs) if len(tracker_inputs) > 0 else np.empty((0, 5))

    # 5-4. 객체 추적기 업데이트
    # tracker.update() 함수는 현재 프레임의 검출 결과를 받아서,
    # 각 객체에 고유한 track_id를 부여한 결과를 리턴.
    tracked_objects = tracker.update(tracker_inputs, frame.shape)

    # 5-5. 각 추적 객체에 대해 화면에 바운딩 박스 및 고유 ID 표시
    for obj in tracked_objects:
        # obj는 보통 dict 또는 객체 형태로, 여기서는 dict 형태로 가정
        # 'bbox' : [x, y, w, h], 'track_id': 고유 ID
        x, y, w, h = obj['bbox']
        track_id = obj['track_id']
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5-6. 최초 신뢰도 조건을 만족한 caller 객체 선택
        # 최초 한 번만 저장하고, 이후에는 갱신하지 않음.
        if first_caller_id is None:
            first_caller_id = track_id
            # 객체 중심 좌표 계산
            center_x = x + w / 2
            center_y = y + h / 2
            # Homography를 통해 이미지 상의 좌표를 지도 좌표로 변환
            point = np.array([[[center_x, center_y]]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(point, H)
            first_caller_world_coord = world_point[0][0]
            print(f"최초 caller (ID: {track_id})의 지도 좌표: {first_caller_world_coord}")

    # 5-7. 최초 caller가 검출된 경우, 그 좌표 정보를 화면에 오버레이
    if first_caller_world_coord is not None:
        cv2.putText(frame, f"First Caller World Coord: {first_caller_world_coord}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 5-8. 결과 프레임 출력
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
