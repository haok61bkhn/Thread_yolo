# Thread_yolo
describe:

  api include :
  
      run detection mutil cam with batch_size
      provide :
        add camera with url(rtsp,path video)
        
  setup : timeout in camera
  
          max_queue in detection in order to auto remove thread camera with message 
          
          
Run:

  api :uvicorn app:app
  test add cam:
      python3 test_request.py
 
