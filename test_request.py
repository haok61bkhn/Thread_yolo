import requests
url = "http://127.0.0.1:8000/add_cam"

data = {'camid':'5','url':"video.mp4"}

req = requests.post(url,  json=data)
print(req.json())
