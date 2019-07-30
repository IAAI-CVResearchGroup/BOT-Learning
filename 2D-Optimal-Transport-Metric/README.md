# 2D-Optimal-Transport-Loss

## Dependencies
The codes are based on Python3.6 and implemented in Tensorflow (Version 1.12.0rc0). Please make sure scikit-learn, opencv, numpy and scipy are also installed successfully.

## Datasets
mnist
cifar-10
cifar-100
Please download cifar-10 and cifar-100 from the official website and put these datasets in the same folder with `run.py`.  

## Codes
(1) `run.py`: train and test our method.  

(2) `inference.py`: include the network architecture (Lenet-5 in this version) and optimal transport loss objective.
