import os
import sys
import random
import math
from glob import glob
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread
import csv

def class2label(classID):
    if classID>0 and classID<9:
        label = '1'
    elif classID > 8 and classID < 15:
        label = '2'
    else:
        label = '0'
    return label

def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)

def get_info(img_path):
    proj = np.fromfile(img_path.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])
    try:
        bbox = np.fromfile(img_path.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except FileNotFoundError:
        return "", "", "", "", ""

    bbox = bbox.reshape([-1, 11])
    
    label = class2label(bbox[0,-2])

    R = rot(bbox[0,0:3])
    t = bbox[0,3:6]
    sz = bbox[0,6:9]
    vert_3D, _ = get_bbox(-sz / 2, sz / 2)
    vert_3D = R @ vert_3D + t[:, np.newaxis]
    vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
    vert_2D = vert_2D / vert_2D[2, :]
    min_x = min(vert_2D[0,:])
    max_x = max(vert_2D[0,:])
    min_y = min(vert_2D[1,:])
    max_y = max(vert_2D[1,:])

    min_x = str(int(max(0, min_x)))
    max_x = str(int(min(1913, max_x)))
    min_y = str(int(max(0, min_y)))
    max_y = str(int(min(1051, max_y)))

    return min_x, min_y, max_x, max_y, label
    
    



def get_bbox( p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([[2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],[7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]], dtype=np.uint8)
    return v, e


if __name__ == "__main__":
    ROOT_PATH = '/home/zixu/Extra_Disk/Dataset/ROB535/all/deploy/'
    train_img_list = glob(ROOT_PATH+'trainval/*/*_image.jpg')
    #test_img_list = glob(ROOT_PATH+'test/*/*_image.jpg')

    # cvs writer
    with open("./rob_train.csv", "w") as csv_file_train:
        writer_train  = csv.writer(csv_file_train, delimiter=',')
        itr=0
        for path in train_img_list:
            itr += 1
            min_x, min_y, max_x, max_y, label = get_info(path)
            writer_train.writerow([path, min_x, min_y, max_x, max_y, label])

    print(itr)
