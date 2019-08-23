# Batch-wise-Optimal-Transport-Metric

## Introduction
  We propose an importance-driven distance metric learning via optimal transport programming from batches of samples, construct a new batch-wise optimal transport loss and combine it into an end-to-end deep metric learning manner. It can emphasize hard samples automatically and lead to significant improvements in convergence.

## Pipeline

![shrec14](shrec14/imgs/framework.jpg?raw=true)

  The proposed batch-wise optimal transport loss is formulated into a deep metric learning framework. Given batches of each modality samples, we use *LeNet-5*, *ResNet-50* and *MVCNN* as *f*<sub>CNN</sub> to extract deep CNN features for 2D images, 2D sketches and 3D shapes, respectively. The metric network *f*<sub>metric</sub> consisting of four fully connected layers, i.e., 4096-2048-512-128 (two FC layers 512-256 for *LeNet-5*) is used to perform dimensionality reduction of the CNN features.  
  The whole framework can be end-to-end trained discriminatively with the new batch-wise optimal transport loss. The highlighted importance-driven distance metrics ***T<sub>ij</sub>M<sub>ij</sub><sup>+</sup>*** and ***T<sub>ij</sub>M<sub>ij</sub><sup>-</sup>*** are used for emphasizing hard positive and negative samples. It jointly learns the semantic embedding metric and deep feature representations for retrieval and classification.

## Multi-modality Retrieval and Classification
(1) 2D Images  
Task: Classification
Datasets: *MNIST* and *CIFAR-10*  
Python Code Framework: Tensorflow  
More details please refer the folder ![2D-Optimal-Transport-Metric](https://github.com/IAAI-CVResearchGroup/Batch-wise-Optimal-Transport-Metric/tree/master/2D-Optimal-Transport-Metric)  

(2) 2D Sketches & 3D Model  
Task: Retrieval  
Datasets: *SHREC13* and *SHREC14*  
Python Code Framework: Tensorflow  
More details please refer the folder ![shrec14](https://github.com/IAAI-CVResearchGroup/Batch-wise-Optimal-Transport-Metric/tree/master/shrec14)  

(3) 3D Shape Recognition  
Task: Classification  
Datasets: *ModelNet10* and *ModelNet40*
Python Code Framework: PyTorch
More details plese refer the folder ![3D-Shape-Recognition]

