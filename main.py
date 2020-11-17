# coding=utf-8
import os
import json
import time
import base64
import logging
from io import StringIO

import cv2
import numpy as np
from flask import Flask
from flask import request

from detector import detect_face, get_detection_model
from recognizer import recognize_attributes, get_recognizer_model
from celeba import CelebA

app = Flask(__name__)
ALLOWED_EXTENSIONS = None
base_path = ''
rec_net = None
det_net = None


# 官方说明中写着放在这里初始化变量可以减少冷启动时间
def initializer(context):
    global rec_net, det_net, base_path, ALLOWED_EXTENSIONS, app
    ALLOWED_EXTENSIONS = {'jpg'}
    base_path = ''

    # 加载模型
    det_model_path = os.environ.get("det_model_path")
    rec_model_path = os.environ.get("rec_model_path")
    rec_net = get_recognizer_model(rec_model_path)
    det_net = get_detection_model(det_model_path)


# 验证后缀名
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 这个不要改
def handler(environ, start_response):
    # 这个路由路径是阿里要求的必写的
    if environ['fc.request_uri'].startswith("/2016-08-15/proxy"):
        from urllib.parse import urlparse
        parsed_tuple = urlparse(environ['fc.request_uri'])
        li = parsed_tuple.path.split('/')
        global base_path
        if not base_path:
            base_path = "/".join(li[0:5])

        context = environ['fc.context']
        environ['HTTP_HOST'] = '{}.{}.fc.aliyuncs.com'.format(
            context.account_id, context.region)
        environ['SCRIPT_NAME'] = base_path + '/'

    return app(environ, start_response)


def get_input(file):
    """
    从文件流中读出图像
    :return: numpy, RGB image
    """
    file_stream = file.stream.read()
    img = cv2.imdecode(np.frombuffer(file_stream, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def process_body(body: str):
    body = json.loads(body)
    return body['image']


def decode_img_from_b64(str_encode: str):
    """
    将上传的图片解码为cv2.imread()的格式
    :param str_encode: 由encode_img编码的字符串
    :return: cv2读取的一帧格式
    """
    img_decode = base64.b64decode(str_encode)
    img_decode = np.fromstring(img_decode, np.uint8)
    img = cv2.imdecode(img_decode, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_logger():
    log_stream = StringIO()
    logger = logging.getLogger('app')
    handler = logging.StreamHandler(log_stream)
    logger.addHandler(handler)
    return logger, log_stream


def crop_face(face_box, image, ratio=5.0):

    resize_h = 218
    resize_w = 178
    h_on_w = resize_h * 1.0 / resize_w

    im_h, im_w, im_c = image.shape
    x1, y1, x2, y2 = face_box
    w, h = (x2 - x1), (y2 - y1)
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2 - int(h * 0.2)
    w2, h2 = int(w * ratio), int(w * h_on_w * ratio)
    x1 = x_center - w2 // 2
    x2 = x_center + w2 // 2
    y1 = y_center - h2 // 2
    y2 = y_center + h2 // 2

    margin_left = 0
    margin_top = 0
    margin_right = 0
    margin_bottom = 0
    if x1 < 0:
        margin_left = - x1
        x1 = 0
    if y1 < 0:
        margin_top = - y1
        y1 = 0
    if x2 > im_w:
        margin_right = x2 - im_w
    if y2 > im_h:
        margin_bottom = y2 - im_h

    fake_img = np.zeros((im_h + margin_top + margin_bottom,
                         im_w + margin_left + margin_right,
                         im_c)).astype(np.uint8)
    fake_img[:, :, 0] = image[:, :, 0].mean()
    fake_img[:, :, 1] = image[:, :, 1].mean()
    fake_img[:, :, 2] = image[:, :, 2].mean()

    fake_img[margin_top:margin_top+im_h, margin_left:margin_left+im_w, :] = image
    face_img = fake_img[y1:y1+h2, x1:x1+w2]
    face_img = cv2.resize(face_img, (resize_w, resize_h))
    return face_img


def main(body):
    logger, log_stream = get_logger()

    image = None
    success = False
    data = None

    try:
        file = process_body(body)
    except Exception:
        file = None
        logger.error('bad request body!', exc_info=True, stack_info=True)

    if file is not None:
        try:
            image = decode_img_from_b64(file)
        except Exception:
            logger.error('bad image!', exc_info=True, stack_info=True)

    if image is not None:
        try:
            face_box = detect_face(image, det_net)
            if face_box is None:
                logger.error('no face detected!')
            else:
                face_image = crop_face(face_box, image, 2.4)
                attr_probs = recognize_attributes(face_image, rec_net)
                res = ['%.4f' % prob for prob in attr_probs]
                data = ' '.join(res)
                success = True
        except Exception:
            logger.error('model runtime error!', exc_info=True, stack_info=True)

    res = {'data': data,
           'success': success,
           'log': log_stream.getvalue()}
    return res


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        # check if the post request has the file part
        # 请求形式是 POST，类型是 form-data，file 字段对应一个图片文件
        body = request.get_json()
        return main(body)
    else:
        return 'request method must be POST'


if __name__ == '__main__':
    app.run()
