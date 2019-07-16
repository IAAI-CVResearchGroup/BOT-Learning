# Batch-wise Optimal Transport Metric for Cross-modality Retrieval

## Introduction
We propose an importance-driven distance metric learning via optimal transport programming from batches of samples, construct a new batch-wise optimal transport loss and combine it into an end-to-end deep metric learning manner. It can emphasize hard samples automatically.

| Criterion | Shrec13 | Shrec14 |
| --- | -- | -- |
| NN | 0.713 |  0.536 |
| FT | 0.728 |  0.564 |
| ST | 0.788 |  0.629 |
| E | 0.366 |  0.305 |
| DCG | 0.818 | 0.712 |
| mAP | 0.754 | 0.591 |

![image](https://github.com/IAAI-CVResearchGroup/Batch-wise-Optimal-Transport-Metric/tree/master/shrec14/imgs)

### Dependencies
The codes are based on Python3.6 and implemented in Tensorflow (Version 1.12.0rc0). For extracting features of sketches and shapes, we also need Caffe framework.

### Datasets
Shrec 13 and Shrec 14.

## Train

### Step 01: Produce 2D images
Each 3D models can be rendered to 20 views of 2D images via Multi-View CNN (MVCNN, [source code](https://github.com/suhangpro/mvcnn)). If you use this method, please consider also citing this [paper](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf).

### Step 02: Extract features from produced 2D images
(1) Extract features using our [pretrained model](https://drive.google.com/drive/folders/1Scc4mwJSWXvnbptj1ZKLjxpTIpZw2toe?usp=sharing) and save them to *.txt:  
`python ./shape/Extract_shape_features.py`  
(2) Save the path of features and labels to `shape_sh.txt`. Please run as  
`python ./shape/Extract_shape_lists.py`

### Step 03: Extract features from multi-view sketches
(1) All sketches are rotated by 20 different angles to extract features using the pretrained model. All features are saved into corresponding *.txt files. Please run  
`python ./sketch/Extract_Features_Train_Sun.py`  
You can choose different mode (train or test) in this code.  
(2) Save the labels and path of features into `sketch_train_sh.txt` and `sketch_test_sh.txt`, respectively. Please run  
`python ./sketch/generateListFile_Sun.py`  

### Step 04: Learn Batch-wise Optimal Transport Metric 
Hyper-parameters can be set by shell file (*.sh) or in `main.py` file. Please run:  
`python ./mixed/main.py`

## Test
In order to computer the criteria in the table, please uncomment `args.phase = 'evaluation'` and change the `self.model_checkpoint_path` in the `./mixed/model.py` to your model path. After that, you can evaluate as:  
`python ./mixed/main.py`  
Our pretrained model can be downloaded from [here](https://drive.google.com/drive/folders/1auvPSuElF_kPlZdMeEHV9sZ6H-2CaDxu?usp=sharing).

## Notes
1. If you have any questions or find any bugs, please let us know: Lin Xu, Han Sun, Zhiyuan Chen {firstname.lastname@horizon.ai}.  
2. With small changes of the datasets and parameters, the code can also be used to train and evaluate the performance on the SHREC13 dataset.
