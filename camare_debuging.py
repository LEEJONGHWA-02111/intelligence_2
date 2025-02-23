import cv2

def find_available_cameras(max_cam_index=10):
    available_cameras = []
    for index in range(max_cam_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

# 실행
available_cams = find_available_cameras()
print("Available cameras:", available_cams)
