# ROB535_Team2
## Task 1:

Install by 
```
cd Task1
pip install -r requirement.txt
python setup.py build_ext --inplace
```

We have trained both RetinaNet and classification layer for you. Download weights from [here](https://drive.google.com/drive/folders/1Qcc4fRP1a3DuhFrSozoDn1LOqHhdGVa4?usp=sharing) and put them into ``Task1/snapshots``. Change the dataset diactory accordingly in file ``rob_gen_data.py``, ``rob_mod_retina2template.py``, ``rob_mod_training.py`` and ``rob_mod_test.py``.

run with

```
python rob_gen_data.py
python rob_mod_retina2template.py
python rob_mod_test.py
```


## Task 2:

Change the directory as comment in the code ``get_depth.m``. We have provide pre-obtained detection bounding box from RetinaNet as ``task2_bbox.csv``

