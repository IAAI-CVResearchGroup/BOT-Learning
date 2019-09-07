# Batch-wise Optimal Transport Metric for 3D Shape Recognition

## Dependecies

Codes are tested on Python 3.6 and PyTorch (1.1.0). Please make sure Python Optimal Transport (POT) library has been installed successfully. The version we used is 0.5.1.  

## Datasets

![*ModelNet40*](supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz)  
ModelNet10（![Baidu Disk](https://pan.baidu.com/s/1hQmC9Z9adjzofeEwrEPCQw), extract code: l8ss）(![Google Drive](https://drive.google.com/file/d/13x8jWCu_BhLImMdGWVt7kWvryKC5wL9x/view?usp=sharing))  

## Implementation
(1) Train  
`python3 mvcnn_optimal_transport.py`

(2) Test  
The command is the same with training. Please set the parameter of `evaluate` in the code to True and load your checkpoint.

## Results
| Criterion | ModelNet40 | ModelNet10 |
| --- | -- | -- |
| Accuracy | 93.1 | 94.2 |
| mAP | 89.2 | 90.0 |

## Notes
The code in this folder is for *ModelNet40* dataset. 

### Parameters for *ModelNet10*
Positive margin = 6.0  
Margin = 8.0  
Learning Rate = 0.001  
Loss Ratio between Softmax and Optimal Transport Loss = 1:1
