import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('') 
    model.train(data='',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=30, 
                workers=4, 
                optimizer='SGD', 
                amp=False, 
                project='runs',
                name='exp',
                )
    