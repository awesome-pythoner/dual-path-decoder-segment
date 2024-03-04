### 这是一个轻量级的语义分割框架

### This is a light semantic segmentation framework

#### > data

这个目录存储了划分后的数据集，在三个子目录中分别用images存储原始数据，masks存储灰度语义掩码，masks_rgb存储调色板模式的语义掩码（非必需）

This directory stores the partitioned datasets, with images for raw data, masks for grayscale semantic masks, and masks_rgb for palette patterns (optional) in three subdirectories

#### > logs

这个目录记录了实验结果，preds目录的子目录images保存了实验结果和ground-truth的对比图，masks保存了模型输出的灰度掩码；test目录保存了评价分割结果的各项指标；train目录保存了训练过程中的损失函数值和精度值

This directory records the experimental results, the subdirectory images of the preds directory stores the comparison chart of the experimental results and ground-truth, masks stores the grayscale mask of the model output, the test directory stores the indicators that evaluate the segmentation results, and the train directory stores the loss function value and accuracy value during the training process

#### > weights

这个目录保存了模型参数

This directory stores metrics of models

#### > models

这个工具包包含了常用的语义分割模型

This toolkit contains commonly used semantic segmentation models

#### > utils

这个工具包包含了常用的工具和模块

This toolkit contains commonly used tools and modules

#### train.py

#### test.py

#### run.py

通过这个脚本文件来进行训练和测试

Use this script file for training and testing
