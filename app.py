import threading
import fastapi
from camera import Camera
from detect import Detection
from typing import Optional
from fastapi import FastAPI
import queue
from pydantic import BaseModel
import uvicorn
detector = Detection()
detector.daemon = True
detector.start()
app = FastAPI()
thread_cams={}

class Item(BaseModel):
    camid: str
    url:str

class Thread(BaseModel):
    id:str

def kill_thread(id):
    thread_cams[id].stop()
    del thread_cams[id]


@app.post("/add_cam/")
async def add_cam(item: Item): 
   try:
    print("receive : ", item)
    if(item.camid in thread_cams.keys()):
        kill_thread(item.camid)
        thread_cams[item.camid]=Camera(item.camid,item.url,detector)
        thread_cams[item.camid].daemon=True
        thread_cams[item.camid].start()
        print("cam ",item.camid," run")
    return {"status":1}
   except:
       return {"status":0}


@app.post("/kill_thread/")
async def kill_thread() : 
   id = random.choice(list(thread_cams.keys()))
   try:
    print("kill thread : ", id)
    kill_thread(id)
    print("kill successful thread " ,item.id)
    return {"status":1}
   except:
       return {"status":0}




   
# if __name__ == '__main__':
#    uvicorn.run("app:app",host='0.0.0.0', port=8000,debug=False,reload=False,workers=0)


