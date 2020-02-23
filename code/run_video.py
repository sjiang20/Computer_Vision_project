import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import datasets
import models
from tensorboardX import SummaryWriter
from utils import transform, util
import datetime
from train import *
from inference import *
import numpy as np
import cv2
from utils import opencv_mser
from scipy.spatial.distance import cdist

VIDEO_INPUT_DIR = './test_input/video1.mp4'
VIDEO_OUTPUT_DIR = './graded_images/video1.mp4'
NET = 'my_net'
WEIGHT_PATH = './svhn/04-18-00-50/Netmy_net_solveradam_epochs800_batchSize1024_lr0.0001_batchnormTrue/model_best.pth.tar'
USE_GPU = True
BATCH_SIZE = 16

def recognize_patches(patches):
    patches = np.asarray(patches)/255.0
    patches = np.transpose(patches, (0, 3, 1, 2))
    tensor = torch.from_numpy(patches)
    tensor = tensor.float()

    trained_weight = torch.load(WEIGHT_PATH)

    model = models.__dict__[NET](trained_weight, isBN = True)
    if USE_GPU==True:
        model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    batches = np.int32(np.ceil(len(patches)*1.0 / BATCH_SIZE))

    softmax = torch.nn.Softmax()
    rst = []
    for i in range(batches):
        s = i * BATCH_SIZE
        e = min((i+1)*BATCH_SIZE, len(patches))
        x = tensor[s:e, :, :, :]
        y = model(x)
        y = softmax(y)
        y = y.data.cpu().numpy()

        tmp_y = np.sort(y)

        conf = tmp_y[:, -1]/tmp_y[:, -2]

        id = np.where(conf < 1000000000)

        maxv = np.argmax(y, axis = 1)

        maxv[id]= 10

        rst.extend(maxv)

        zx = 0
    return rst

def non_maximum_suppression(bbox, digits, id_has_digit):
    bbox_digits = []
    digits_digits = []

    for id in id_has_digit[0]:
        bbox_digits.append(bbox[id])
        digits_digits.append(digits[id])

    corners = []

    for bbox in bbox_digits:
        p = [bbox[0], bbox[2]]
        corners.append(p)

    if corners==[]:
        return bbox, digits

    corners = np.asarray(corners)
    dist = cdist(corners, corners)

    bbox_new = []
    digits_new = []
    myMap = np.zeros(len(corners))

    for i in range(len(corners)):
        if myMap[i]==1:
            continue

        d = dist[i, :]
        qualified_id = np.where(d<50)
        bbox_new.append(bbox_digits[i])
        digits_new.append(digits_digits[i])
        myMap[qualified_id] = 1

    return bbox_new, digits_new

def video_frame_generator(filename):
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None

def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def main():
    fps = 40
    image_gen = video_frame_generator(VIDEO_INPUT_DIR)

    image = image_gen.__next__()
    h, w, d = image.shape
    video_out = mp4_video_writer(VIDEO_OUTPUT_DIR, (w, h), fps)

    frame_num = 0

    count = 0

    while image is not None:
        print("Processing frame {}".format(count))

        patches, bbox, img_proposed_ragions = opencv_mser.extract_candidate_regions(image)
        if len(patches)==0:
            image = image_gen.__next__()
            count = count + 1
            continue

        digits = recognize_patches(patches)
        id_has_digit = np.where(np.asarray(digits) != 10)

        bbox_new, digits_new = non_maximum_suppression(bbox, digits, id_has_digit)

        rst_image = image.copy()

        for id in range(len(bbox_new)):
            cur_bbox = bbox_new[id]
            cur_digit = digits_new[id]

            xmin = cur_bbox[0]
            xmax = cur_bbox[1]
            ymin = cur_bbox[2]
            ymax = cur_bbox[3]

            cv2.rectangle(rst_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (xmin, ymax)
            fontScale = 0.5
            fontColor = (255, 0, 0)
            lineType = 1

            cv2.putText(rst_image, str(cur_digit), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        video_out.write(np.uint8(rst_image))
        image = image_gen.__next__()
        count = count + 1
    video_out.release()


if __name__ == '__main__':
    main()