#USB 카메라를 이용하여 지속적으로 이미지를 캡처하는 기능
#사용자가 파일 이름을 설정할 수 있으며, 1초마다 자동으로 이미지를 저장

import cv2
import os
import time

save_directory = "img_capture_1"  #save direectory is set to be under the current directory

def capture_image():

    os.makedirs(save_directory, exist_ok=True)

    file_prefix = input("Enter a file prefix to use : ")
    file_prefix = f'{file_prefix}_'
    print(file_prefix)
    
    image_count = 0
    # cap = cv2.VideoCapture(0)   #PC Camera
    cap = cv2.VideoCapture(1)   #USB Camera
    
    while True:
        ret, frame = cap.read()
        
        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        else:
            file_name = f'{save_directory}/{file_prefix}img_{image_count}.jpg'
            cv2.imwrite(file_name, frame)
            print(f"Image saved. name:{file_name}")
            image_count += 1

            # Wait for 1 second
            time.sleep(1)
        
    cap.release()
    cv2.destroyAllWindows()

def main():
    capture_image()

if __name__ == "__main__":
    main()

