# Tree Canopy Detection Using Swin-Unet

This repository contains my implementation and modifications of the Swin-Unet architecture for the task of **tree canopy detection** using my own dataset.  
The project adapts the original Swin-Unet model (designed for medical image segmentation) to a remote-sensing / environmental application.

## About This Project
- Based on the original Swin-Unet implementation by Hu Cao et al.  
- Modified to support tree canopy segmentation datasets  
- Includes changes in:
  - dataset loading pipeline  
  - configuration files  
  - training hyperparameters  
  - preprocessing scripts  
  - evaluation metrics (IoU, pixel accuracy, etc.)  

## Reference to Original Work
This project is **built on top of the following repository**:

**Swin-Unet: https://github.com/HuCaoFighting/Swin-Unet**

Please refer to the original paper for details about the architecture:

```bibtex
@InProceedings{swinunet,
author    = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
title     = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
booktitle = {Proceedings of the European Conference on Computer Vision Workshops (ECCVW)},
year      = {2022}
}


