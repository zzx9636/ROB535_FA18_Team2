import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from keras.optimizers import SGD
import csv
import tensorflow as tf
# Generate dummy data
import numpy as np


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, labels, batch_size=4, n_classes=3, training=True, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.training = training
        self.labels = labels
        self.list_IDs = path
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        y = np.empty((self.batch_size), dtype=int)

        # Find list of IDs
        list_IDs_temp = []
        for i, k in enumerate(indexes):
            list_IDs_temp.append(self.list_IDs[k])
            y[i] = self.labels[k] 

        # Generate data
        X = self.__data_generation(list_IDs_temp)
        y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 256*256*3))
        

        # Generate data
        for i, cur_path in enumerate(list_IDs_temp):
            npy_path = path2savepath(cur_path, self.training)
            template = np.load(npy_path)
            template_flat = template.flatten()
            X[i,:] = template_flat

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

def simple_classification():
    model = keras.Sequential()
    model.add(Dense(512, input_shape=(256*256*3,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

if __name__ == "__main__":
    ROOT_PATH = "/home/zixu/Extra_Disk/Dataset/ROB535/all/deploy"
    train_path_list = []
    train_label_list = []
    with open('./rob_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            train_path_list.append(row[0])
            train_label_list.append(int(row[-1]))

    eval_path_list = []
    eval_label_list = []

    with open('./rob_eval.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                path = ROOT_PATH+"/test/"+row[0]+"_image.jpg"
                eval_path_list.append(path)
                eval_label_list.append(int(row[-1]))
        
    model = simple_classification()
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])
    train_gen = DataGenerator(train_path_list, train_label_list, batch_size = 16)
    val_gen = DataGenerator(eval_path_list, eval_label_list, batch_size=16, training=False)
    model.fit_generator(generator=train_gen, epochs=5, steps_per_epoch=450,
                            validation_data=val_gen, validation_steps=150)
    model.save("snapshots/template_weight.h5")
