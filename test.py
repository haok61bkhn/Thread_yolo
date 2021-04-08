from yolov5_original.detect import YOLOV5
import cv2
import glob
detector=YOLOV5()
imgs=[]
for path in glob.glob("test/*"):
    imgs.append(cv2.imread(path))
    
res_boxes,res_classes,res_confs=detector.detect(imgs)
print(res_classes)