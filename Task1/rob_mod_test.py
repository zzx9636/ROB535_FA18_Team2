import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from keras.optimizers import SGD
import csv
import tensorflow as tf
# Generate dummy data
import numpy as np

model = keras.models.load_model("snapshots/template_weight.h5")
print(model.summary())

def data_generation(path):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((1, 256*256*3))
    # Generate data
    npy_path = path2savepath(path, False)
    template = np.load(npy_path)
    template_flat = template.flatten()
    X[0,:] = template_flat
    return X

def path2savepath(path, if_train):
    path_split = path.split('/')
    folder_name = path_split[-2]
    img_name = path_split[-1].split('.')[0]
    if if_train:
        save_path = './rob_box/train/'+folder_name+'_'+img_name+'.npy'
    else:
        save_path = './rob_box/eval/'+folder_name+'_'+img_name+'.npy'
    return save_path

def inference(path):
    template = data_generation(path)
    label = model.predict_on_batch(template)
    idx = np.argmax(label[0])
    #print(label[0])
    #print(idx)
    return idx

if __name__ == "__main__":
    ROOT_PATH = "/home/zixu/Extra_Disk/Dataset/ROB535/all/deploy"
    eval_path_list=[]
    eval_name_list=[]
    with open('./rob_eval.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                eval_name_list.append(row[0])
                path = ROOT_PATH+"/test/"+row[0]+"_image.jpg"
                eval_path_list.append(path)

    with open("./test_inf.csv", "w") as csv_file:
        writer  = csv.writer(csv_file, delimiter=',')
        writer.writerow(['guid/image','label'])
        for path, name in zip(eval_path_list, eval_name_list):
            out_label=inference(path)
            #out_label=0
            writer.writerow([name,str(out_label)])
    
