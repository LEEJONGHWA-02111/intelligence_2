import cv2

# 마우스 이벤트 콜백 함수 정의
def click_event(event, x, y, flags, param):
    # 왼쪽 버튼 클릭 시
    if event == cv2.EVENT_LBUTTONDOWN:
        print("픽셀 좌표: ({}, {})".format(x, y))
        # 클릭한 위치에 작은 원을 그리고 좌표를 이미지에 표시 (옵션)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, f"({x},{y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)

# 이미지를 파일에서 읽어오기 (이미지 경로를 실제 파일 경로로 변경)
img = cv2.imread('/home/leejonghwa/Downloads/inteligence2/map.pgm')
if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

# 이미지 창 생성 및 이미지 표시
cv2.imshow('image', img)
# 마우스 콜백 함수 등록 (창 이름과 함수 이름을 지정)
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
