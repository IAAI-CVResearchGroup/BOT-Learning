# Batch-wise-Optimal-Transport-Metric

## Introduction
We propose an importance-driven distance metric learning via optimal transport programming from batches of samples, construct a new batch-wise optimal transport loss and combine it into an end-to-end deep metric learning manner. It can emphasize hard samples automatically and lead to significant improvements in convergence.

## Pipeline

![shrec14](shrec14/imgs/framework.jpg?raw=true)

The proposed batch-wise optimal transport loss is formulated into a deep metric learning framework. Given batches of each modality samples, we use *LeNet-5*, *ResNet-50* and *MVCNN* as f<sup>1</sup>
