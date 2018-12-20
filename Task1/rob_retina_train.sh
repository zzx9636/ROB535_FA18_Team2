#!/bin/sh
python keras_retinanet/bin/train.py --snapshot=snapshots/resnet50_csv_01.h5 --random-transform --steps=7500 --epochs=1 csv ./rob_train.csv ./rob_class_id.csv
