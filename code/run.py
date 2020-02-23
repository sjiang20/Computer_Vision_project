import os

import cv2
import numpy as np
import torch.nn.parallel
import torch.optim
import torch.utils.data
from scipy.spatial.distance import cdist

import models
from inference import *
from utils import opencv_mser

IMG_INPUT_DIR = './test_input'
IMG_OUTPUT_DIR = './graded_images'
NET = 'my_net'
WEIGHT_PATH = './svhn/04-18-00-50/Netmy_net_solveradam_epochs800_batchSize1024_lr0.0001_batchnormTrue/model_best.pth.tar'
USE_GPU = False
BATCH_SIZE = 16

def recognize_patches(patches):
    patches = np.asarray(patches)/255.0
    patches = np.transpose(patches, (0, 3, 1, 2))
    tensor = torch.from_numpy(patches)
    tensor = tensor.float()

    trained_weight = torch.load(WEIGHT_PATH, map_location='cpu')

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

    corners = np.asarray(corners)
    dist = cdist(corners, corners)

    bbox_new = []
    digits_new = []
    myMap = np.zeros(len(corners))

    for i in range(len(corners)):
        if myMap[i]==1:
            continue

        d = dist[i, :]
        qualified_id = np.where(d<10)
        bbox_new.append(bbox_digits[i])
        digits_new.append(digits_digits[i])
        myMap[qualified_id] = 1

    return bbox_new, digits_new

def main():
    img_paths = [os.path.join(IMG_INPUT_DIR, f) for f in os.listdir(IMG_INPUT_DIR) if f.endswith('.jpg')]

    count = 0
    for img_path in img_paths:
        img = cv2.imread(img_path)
        patches, bbox, img_proposed_ragions = opencv_mser.extract_candidate_regions(img)
        # cv2.imshow("text only", img_proposed_ragions)
        # cv2.waitKey(0)
        digits = recognize_patches(patches)
        id_has_digit = np.where(np.asarray(digits)!=10)

        bbox_new, digits_new = non_maximum_suppression(bbox, digits, id_has_digit)

        for id in range(len(bbox_new)):
            cur_bbox = bbox_new[id]
            cur_digit = digits_new[id]

            xmin = cur_bbox[0]
            xmax = cur_bbox[1]
            ymin = cur_bbox[2]
            ymax = cur_bbox[3]

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (xmin, ymax)
            fontScale = 0.5
            fontColor = (255, 0, 0)
            lineType = 1

            cv2.putText(img, str(cur_digit), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        count = count + 1
        img_out_path = os.path.join(IMG_OUTPUT_DIR, str(count)+'.png')
        cv2.imwrite(img_out_path, img)

        # cv2.imshow("text only", img)
        # cv2.waitKey(0)

if __name__ == '__main__':
    main()