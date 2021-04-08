import threading
import time
import cv2
class Camera(threading.Thread):
    def __init__(self,id,url,thread,timeout=3):
        self.id=id
        self.status=1
        self.thread=thread
        self.url=url
        self.timeout=timeout
        super(Camera, self).__init__()
        self._stop_event = threading.Event()


    def stop(self):
        self._stop_event.set()
    
    def run(self):
        cam=cv2.VideoCapture(self.url)
        while True:
            _,image=cam.read()
            if image is None:
                t1=time.time()
                while image is not None :
                    _,image = cam.read()
                    if(time.time()-t1>self.timeout):
                        #todo
                        self.stop()
            self.thread.add_image(image,self.id)
            time.sleep(1)