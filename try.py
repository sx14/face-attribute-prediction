import requests
import base64
import json

# url = "http://localhost:8000/2016-08-15/proxy/face_attr/face_attr"
url = 'https://1889584630546789.cn-shanghai.fc.aliyuncs.com/2016-08-15/proxy/face_attr/face_attr/'
with open("imgs/1.jpg", 'rb') as f:
    base64_data = base64.b64encode(f.read())
data = {"image": base64_data.decode()}
res = requests.post(url=url, json=json.dumps(data))
print(res.text)
