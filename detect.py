from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('runs/detect/n-NAM-ada-IBE-Slide/weights/best.pt')
    model.predict(source="picture", imgsz=640, device=0, save=True)