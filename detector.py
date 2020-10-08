# coding=utf-8
import argparse
import os
import time
from math import ceil

import cv2
import numpy as np
from cv2 import dnn

parser = argparse.ArgumentParser()
parser.add_argument('--caffe_prototxt_path', default="detection/model/RFB-320/RFB-320.prototxt", type=str, help='caffe_prototxt_path')
parser.add_argument('--caffe_model_path', default="detection/model/RFB-320/RFB-320.caffemodel", type=str, help='caffe_model_path')
parser.add_argument('--onnx_path', default="detection/onnx/onnx/version-RFB-320_simplified.onnx", type=str, help='onnx version')
parser.add_argument('--input_size', default="240,320", type=str, help='define network input size,format: width,height')
parser.add_argument('--threshold', default=0.0, type=float, help='score threshold')
parser.add_argument('--imgs_path', default="detection/imgs", type=str, help='imgs dir')
parser.add_argument('--results_path', default="detection/results", type=str, help='results dir')

threshold = 0.5
input_size = [240, 320]
image_std = 128.0
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
strides = [8.0, 16.0, 32.0, 64.0]


def get_detection_model(weight_path):
    net = dnn.readNetFromONNX(weight_path)  # onnx version
    return net


def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    return priors


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    return np.clip(priors, 0.0, 1.0)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)


def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                           locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)


def detect_face(image, net):
    """
    人脸检测
    :param: numpy, RGB image
    :return: list, best face box [x1, y1, x2, y2]
    """
    # net = dnn.readNetFromONNX(args.onnx_path)  # onnx version
    witdh = input_size[0]
    height = input_size[1]
    priors = define_img_size(input_size)

    rect = cv2.resize(image, (witdh, height))
    net.setInput(dnn.blobFromImage(rect, 1 / image_std, (witdh, height), 127))
    boxes, scores = net.forward(["boxes", "scores"])
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
    boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
    boxes = center_form_to_corner_form(boxes)
    boxes, labels, probs = predict(image.shape[1], image.shape[0], scores, boxes, threshold)
    if boxes.shape[0] == 0:
        return None
    else:
        best_id = np.argmax(probs)
        best_box = boxes[best_id]
        best_box = [max(0, best_box[0]),
                    max(0, best_box[1]),
                    min(image.shape[1], best_box[2]),
                    min(image.shape[0], best_box[3])]
        return best_box

