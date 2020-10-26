# coding=utf-8
import os

import cv2
import numpy as np
from tqdm import tqdm

from detector import detect_face, get_detection_model
from recognizer import recognize_attributes, get_recognizer_model
from celeba import CelebA

base_path = ''
rec_net = None
det_net = None
logger = None


def initialize(use_gpu=False):
    global rec_net, det_net, attr_names, logger
    det_model_path = 'detection/onnx/onnx/version-RFB-320_simplified.onnx'
    rec_model_path = 'checkpoints/resnet50/model_best_fix.pth.tar'
    det_net = get_detection_model(det_model_path)
    rec_net = get_recognizer_model(rec_model_path)
    attr_names = CelebA.attr_names()
    if use_gpu:
        rec_net = rec_net.cuda()

    logger = open('log_face_attribute.txt', 'w')


def get_input(img_root):
    """
    获得输入视频帧
    :return: [numpy], RGB image list
    """
    imgs = []
    fids = []
    for img_id in sorted(os.listdir(img_root)):
        img_path = os.path.join(img_root, img_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        fids.append(img_id)
    return fids, imgs


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


def main(video_list_path, res_root, use_gpu=False):
    global logger

    with open(video_list_path) as f:
        video_list = [line.strip().split()[1] for line in f.readlines()]

    if not os.path.exists(res_root):
        os.makedirs(res_root)

    for frame_dir in tqdm(video_list):
        video_frame_root = frame_dir
        fids, images = get_input(video_frame_root)

        vid = frame_dir.split('/')[-1]
        res_path = os.path.join(res_root, vid + '.txt')
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                lines = f.readlines()
            if len(lines) == len(fids):
                continue

        results = []
        for i, image in enumerate(images):
            face_box = detect_face(image, det_net)
            if face_box is None:
                res = ['NaN'] * len(attr_names)
            else:
                face_image = crop_face(face_box, image, 2.4)
                attr_probs = recognize_attributes(face_image, rec_net, use_gpu)
                res = [attr_probs[i] for i in range(len(attr_probs))]
                res = ['%.4f' % scr for scr in res]
            results.append(res)

        with open(res_path, 'w') as f:
            for i in range(len(fids)):
                line = '%s %s\n' % (fids[i].split('.')[0], ' '.join(results[i]))
                f.writelines(line)

            logger.write('%s\n' % vid)
            logger.flush()

        logger.write('%s done!' % video_list_path)
        logger.flush()
        logger.close()


if __name__ == '__main__':
    video_list_path = 'test/data/4.0/xxyb-frame.txt'
    res_root = 'test/res/4.0/face_attribute'
    initialize(use_gpu=True)
    main(video_list_path, res_root, use_gpu=True)
