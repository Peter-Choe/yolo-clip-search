import requests
import os ,sys
import cv2
import numpy as np
import random
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.config import RUN_ID



API_URL = "http://localhost:5000/detect/"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/image_search

# 입력 디렉토리
input_dir = os.path.join(ROOT_DIR, "datasets/coco_subset/version_3/images/test")

# 처리할 이미지 수
N = 50


# 현재 시각 기반으로 고유 output 디렉토리 생성
timestamp = datetime.now().strftime("%H%M%S")
output_dir = os.path.join(ROOT_DIR, f"test/output_images/run_{RUN_ID}_{timestamp}_{N}")
os.makedirs(output_dir, exist_ok=True)

# 이미지 리스트 가져오기
all_images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
sample_images = random.sample(all_images, min(N, len(all_images)))

print(f"Running detection on {len(sample_images)} images...\n")
print(f"Results will be saved to: {output_dir}\n")

for image_name in sample_images:
    image_path = os.path.join(input_dir, image_name)
    output_path = os.path.join(output_dir, f"detected_{image_name}")

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"[SKIP] Failed to read {image_name}")
        continue

    # API 요청
    with open(image_path, "rb") as f:
        files = {"file": (image_name, f, "image/jpeg")}
        response = requests.post(API_URL, files=files)
        print(f"=============================================================================================\n")
        print(f"{response.json()}\n\n")

    # 결과 처리
    if response.status_code == 200:
        detections = response.json()["detections"]
        print(f"[INFO] {image_name} - Detected {len(detections)} objects")
        print(f' Detected objects: {", ".join([det["name"] for det in detections])}\n')

        for det in detections:
            xmin = int(det["xmin"])
            ymin = int(det["ymin"])
            xmax = int(det["xmax"])
            ymax = int(det["ymax"])
            label = det["name"]
            conf = det["confidence"]

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(image, text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(output_path, image)
        print(f"[SAVED] {output_path}\n")

    else:
        print(f"[ERROR] {image_name} - Detection failed with status {response.status_code}")
        print(response.text)
