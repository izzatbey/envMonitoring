import cv2
import os
import re
import time
from dotenv import load_dotenv

load_dotenv

def get_max_counter(dir):
    try:
        files = os.listdir(dir)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return 0

    max_counter = 0
    regex = re.compile(r'image_(\d+)\.jpg')

    for file in files:
        matches = regex.match(file)
        if matches:
            try:
                counter = int(matches.group(1))
                if counter > max_counter:
                    max_counter = counter
            except ValueError:
                continue

    return max_counter

def capture_images():
    img_folder = os.getenv("IMG_FOLDER")
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error opening video capture device")
        return

    counter = get_max_counter(img_folder) + 1

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: cannot read device")
            return
        if frame is None:
            continue

        filename = f"{img_folder}/image_{counter}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved ", filename)

        counter += 1

        time.sleep(2)

if __name__ == "__main__":
    capture_images()
