from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('runs/detect/n/weights/best.pt')
    model.val(imgsz=640, batch=1, device=0)

