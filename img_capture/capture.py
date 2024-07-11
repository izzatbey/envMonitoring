import cv2
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

def capture_images():
    img_folder = os.getenv("IMG_FOLDER")
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error opening video capture device")
        return

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: cannot read device")
            return
        if frame is None:
            continue
        
        train_data_path = os.getenv("TRAIN_DATA_DIRECTORY")
        filename = f"{train_data_path}/camera/image_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved ", filename)

        time.sleep(2)

if __name__ == "__main__":
    capture_images()
