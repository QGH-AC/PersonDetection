import os
from ultralytics import YOLO
import cv2

def detect_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 720))
    results = model(img)
    for result in results:
        boxes = result.boxes
        for box, cls in zip(boxes, result.boxes.cls):
            class_id = int(cls)
            if model.names[class_id] == "person":
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = box.conf[0]
                print(f"在({x1}, {y1}), ({x2}, {y2})处检测到人物,准确率为: {conf:.2f}")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Person: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)

def detect_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (640, 720))
        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box, cls in zip(boxes, result.boxes.cls):
                class_id = int(cls)
                if model.names[class_id] == "person":
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = box.conf[0]
                    print(f"在({x1}, {y1}), ({x2}, {y2})处检测到人物,准确率为: {conf:.2f}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Result", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
def detect_images_in_folder(model, folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(root, file)
                detect_image(model, image_path)

model = YOLO("yolo11n.pt")

image_path = "test.jpg"
print("检测单张图片")
detect_image(model, image_path)

folder_path = "media/images"
print("遍历图片文件夹检测")
detect_images_in_folder(model, folder_path)

video_path = "media/videos/test.mp4"
print("检测视频文件")
detect_video(model, video_path)