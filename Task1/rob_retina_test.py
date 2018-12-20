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

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
labels_to_names = {0: '0', 1: '1', 2:'2'}

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
model_path = os.path.join('./', 'snapshots','resnet50_csv_02_t.h5' )

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

print(model.summary())

# load label to names mapping for visualization purposes

def inference_img(path, show=False):
    # load image
    image = read_image_bgr(path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    # correct for image scale
    boxes /= scale
    if show:
        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break 
            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
            
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()
        #print(scores)
    idx = np.argmax(scores[0])
    #print(idx)
    out_label = labels[0][idx]
    if out_label==-1:
        out_label=0
    #print(out_label)
    return out_label

ROOT_PATH = '/home/zixu/Extra_Disk/Dataset/ROB535/all/deploy/'
test_img_list = glob(ROOT_PATH+'test/*/*_image.jpg')
with open("../test_inf.csv", "w") as csv_file:
    writer  = csv.writer(csv_file, delimiter=',')
    writer.writerow(['guid/image','label'])
    for path in test_img_list:
        out_label = inference_img(path, show=False)
        image_path_split = path.split("/")
        image_path = image_path_split[-2]+"/"+image_path_split[-1].split("_")[0]
        writer.writerow([image_path,str(out_label)])


