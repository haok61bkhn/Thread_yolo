import sys
import os
sys.path.insert(0, "Detection")

from norfair import  Tracker, Video, draw_tracked_objects,Detection
import cv2
import numpy as np
from detect import YOLOV5
import uuid
from license_plate import LicensePlate
import cv2
from PIL import Image, ImageDraw
import threading
import time

FINAL_LINE_COLOR = (255, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX

class Parking(threading.Thread):
    def __init__(self,polygon,width,height,url, idcam):
        self.lock = threading.RLock()
        self.polygon = polygon
        self.url=url
        self.width=width
        self.height=height
        self.detector=YOLOV5(classes=["car"])
        self.tracker = Tracker(distance_function=self.euclidean_distance, distance_threshold=50,initialization_delay=1)
        self.res={}
        self.lpn = LicensePlate()
        self.update=[]
        self.idcamera = idcam
        self.create_mask()
        super(Parking, self).__init__()
        self._stop_event = threading.Event()


    def stop(self):
        self._stop_event.set()

    def create_mask(self):
        img = Image.new('L', (self.width,self.height), 0)
        ImageDraw.Draw(img).polygon(self.polygon, outline=1, fill=1)
        self.mask = np.array(img)
    def set_mask(self,polygon):
        with self.lock:
            self.polygon=polygon
            self.create_mask()


    def euclidean_distance(self,detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

        
    def process_object_tracking(self,tracked_objects,frame):
        # try:
        for obj in tracked_objects:
                id=obj.id
                for point, live in zip(obj.estimate, obj.live_points):
                    if live:
                        x,y,w,h=(obj.last_detection.data)
                        
                        x-=w//2
                        y-=h//2
                        if(id not in self.res.keys()):
                            vhc_img  = frame[y:y+h,x:x+w]
                            lpn_det = self.lpn.detect_lpn(vhc_img)
                            lpn_predict=""
                            if(lpn_det[1]!=0):
                                lpn_predict=self.lpn.predict(lpn_det[0])

                            self.res[id]=[x,y,w,h,vhc_img,lpn_predict,lpn_det[1]]
                        else:
                                vhc_img  = frame[y:y+h,x:x+w]
                           
                                lpn_det = self.lpn.detect_lpn(vhc_img)
                                if(lpn_det[1]>self.res[id][6]):
                                    lpn_predict=self.lpn.predict(lpn_det[0])
                                
                                    self.res[id]=[x,y,w,h,vhc_img,lpn_predict,lpn_det[1]]

                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,125),2)
                        # print(self.res[id][5])
                        # cv2.putText(frame,self.res[id][5],(x+w//2,y+h//2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                        print("parking : ","video_id ",self.url," id_object :",id," license plate : ",self.res[id][5])
        
        # except Exception as e:
        #     print(e)
        #     pass
        return frame
                
    

                    

    def run(self):
        cam=cv2.VideoCapture(self.url)
        while(cam.isOpened()):
            _,frame=cam.read()
            box_detects,ims,classes=self.detector.detect(frame)  # box_detects.append([(left+right)//2,(top+bottom)//2, right-left,bottom-top])
           
            detections = [Detection(np.array([box[0],box[1]]),data=np.array([box[0],box[1],box[2],box[3]])) for box in box_detects if (self.mask[box[1]][box[0]]) ]
            
            tracked_objects = self.tracker.update(detections=detections)
            frame=self.process_object_tracking(tracked_objects,frame)
            # frame=cv2.polylines(frame, np.array([self.polygon]), False, FINAL_LINE_COLOR, 1)
            
            # cv2.imshow('frame',frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    
if __name__ == '__main__':
    image=cv2.imread("supervision.png")
    height,width=image.shape[:2]
    X=Parking(url="video.mp4",polygon = [(428, 354), (1271, 321), (1661, 453), (1231, 852), (638, 864), (396, 507), (421, 364)],width=width,height=height)
    X.start()
    # print(len(X.process_video()))
    time.sleep(5)
    X.set_mask([(919, 361), (1452, 275), (1749, 391), (1297, 770), (919, 363)])