
import threading,queue
import time
from yolov5_original.detect import YOLOV5

class Detection(threading.Thread):
    def __init__(self,batch_size=5,max_queue_size=100):
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        self.batch_size=batch_size
        self.detector=YOLOV5()
        self.url_kill="http://127.0.0.1:8000/kill_thread"
        self.max_queue_size
        super(Detection, self).__init__()
        self._stop_event = threading.Event()
        



    def add_image(self,image,id):
        with self.lock:
            self.queue.put({"image":image,"id":id})

    def stop(self):
        self._stop_event.set()

    def process_queue(self): 
        items=[]
        while(self.queue.qsize()>0):
            items.append(self.queue.get()['image'])
            if(len(items)==self.batch_size):
                break
        if(len(items)>0):
            res_boxes,res_classes,res_confs=self.detector.detect(items)
            print(res_classes)

       
        
        print("len items :",len(items))

    def kill_thread(self):
        req = requests.post(url)
        
    
    def run(self):
        while True:
            # with self.lock:
                print("number images :",self.queue.qsize())
                time.sleep(1)
                self.process_queue()
                if(self.queue.qsize()>self.max_queue_size-5):
                    #warning
                    print("kill")

                