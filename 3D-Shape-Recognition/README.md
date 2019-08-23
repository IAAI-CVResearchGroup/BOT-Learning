# Batch-wise Optimal Transport Metric for 3D Shape Recognition

## Dependecies

Codes are tested on Python 3.6 and PyTorch (1.1.0). Please make sure Python Optimal Transport (POT) library has been installed successfully. The version we used is 0.5.1.  

## Datasets

![*ModelNet40*](supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz)  
![*ModelNet10*]

## Implementation
(1) Train  
`python3 mvcnn_optimal_transport.py`

(2) Test  
The command is the same with training. Please set the parameter of `evaluate` in the code to True and load your checkpoint.

## Results
| Criterion | ModelNet40 | ModelNet10 |
| --- | -- | -- |
| Accuracy | 0.931 | 0.937 |
| mAP | 0.889 | 0.875 |

## Notes
The code in this folder is for *ModelNet40* dataset. With small changes of the datasets and parameters, the code can also be used to train and evaluate the performance on the *ModelNet10* dataset.
