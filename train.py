# -!- coding: utf-8 -!-
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("config/n-NAM-IBE.yaml")  # 用于迁移训练的权重文件路径

    results = model.train(data="HGI30.yaml", imgsz=640, epochs=300, batch=32, device=0, workers=8)