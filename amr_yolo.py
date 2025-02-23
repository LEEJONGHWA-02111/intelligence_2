# YOLOv8을 사용하여 객체 탐지(Object Detection) 및 추적(Tracking)을 수행하는 코드

from ultralytics import YOLO
import cv2

model = YOLO('/home/rokey8/rokey_ws/yolo/yolov8n.pt')

# TensorRT 모델(yolov8n.engine)을 직접 로드하여 객체 탐지 수행
trt_model = YOLO('/home/rokey8/rokey_ws/yolo/yolov8n.engine', task="detect")

img = "/home/rokey8/rokey_ws/yolo/bus.jpg"

# Load the JPG image
frame = cv2.imread(img)  # Reads in BGR format

# Check if the image was loaded successfully
if frame is None:
    print("Error loading image")
else:
    print("Image shape:", frame.shape)  # (height, width, channels)

# YOLOv8 PyTorch 모델(yolov8n.pt)을 사용하여 객체 탐지 수행
print("\n***standard inference")
results = model(frame)

# TensorRT 엔진을 사용하여 객체 탐지 실행(YOLOv8 모델보다 추론 속도가 훨씬 빠름)
print("\n***TensorFlow RT inference")
results = trt_model(frame)

# YOLOv8 + TensorRT 기반으로 객체 추적 실행
print("\n***TensorFlow RT inference Tracking")
# ByteTrack(bytetrack.yaml) 알고리즘을 사용하여 객체 ID를 추적
results = trt_model.track(source=frame, show=False, tracker='bytetrack.yaml')

# results에서 객체의 **Bounding Box(경계 상자), 신뢰도(conf), 클래스 ID(cls), 추적 ID(track_id)**를 가져옴
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
        track_id = int(box.id[0]) if box.id is not None else None  # Tracking ID
        conf = float(box.conf[0])  # Confidence score
        cls = int(box.cls[0])  # Class ID
        
        # Get class name (optional)
        class_name = model.names[cls] if cls in model.names else "Unknown"

        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Display tracking ID, confidence, and class name
        label = f"ID: {track_id} {class_name} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


cv2.imshow('Tracked Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



