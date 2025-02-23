import cv2

cap = cv2.VideoCapture(0)  # 웹캠 열기
if not cap.isOpened():
    print("Error: Cannot open camera")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Cannot read frame")
cap.release()
