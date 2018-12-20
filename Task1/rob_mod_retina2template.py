import os, sys

import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from glob import glob
import csv
import tensorflow as tf
from keras.objectives import categorical_crossentropy

from keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from keras.layers.core import K



labels_to_names = {0: '0', 1: '1', 2:'2'}

#height = 733
#width = 1333

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path_train = os.path.join('./', 'snapshots','resnet50_csv_01.h5' )
model_path_inf = os.path.join('./', 'snapshots','resnet50_csv_02_t.h5' )
ROOT_PATH = '/home/zixu/Extra_Disk/Dataset/ROB535/all/deploy/'
test_img_list = glob(ROOT_PATH+'test/*/*_image.jpg')

# load retinanet model
#model_train = models.load_model(model_path_train, backbone_name='resnet50')
model_inf = models.load_model(model_path_inf, backbone_name='resnet50')

def path2savepath(path, if_train):
    path_split = path.split('/')
    folder_name = path_split[-2]
    img_name = path_split[-1].split('.')[0]
    if if_train:
        save_path = './rob_box/train/'+folder_name+'_'+img_name+'.npy'
    else:
        save_path = './rob_box/eval/'+folder_name+'_'+img_name+'.npy'
    return save_path

def getdata(path, if_train):
    save_path = path2savepath(path, if_train)
    image = read_image_bgr(path)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model_inf.predict_on_batch(np.expand_dims(image,axis=0))
    template = np.zeros([256, 256, 3])
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score<0.1:
            break
        x1 = int(np.ceil(box[0]/1333.0*256))
        y1 = int(np.ceil(box[1]/733.0*256))
        x2 = int(np.floor(box[2]/1333.0*256))
        y2 = int(np.floor(box[3]/733.0*256))
        template[y1:y2, x1:x2, label] = score
    np.save(save_path,template)
    
if __name__ == "__main__":
    path_list = []
    train=False
    ROOT_PATH = "/home/zixu/Extra_Disk/Dataset/ROB535/all/deploy"
    if train:
        with open('./rob_train.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                path_list.append(row[0])
    else:
        with open('./rob_eval.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                path = ROOT_PATH+"/test/"+row[0]+"_image.jpg"
                path_list.append(path)

    for path in path_list:
        getdata(path,train)

   








