# Tree Canopy Detection Using Swin-Unet

This repository contains my implementation and modifications of the Swin-Unet architecture for the task of **tree canopy detection**.  
The project was developed as part of the **Solafune Tree Canopy Detection Competition**, where the goal is to segment tree canopy coverage from aerial imagery.

## About the Competition Dataset
This work uses the dataset from the following competition:

**Solafune Tree Canopy Detection Challenge**  
https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?menu=about&tab=  

**Dataset description:**  
- High-resolution aerial images  
- Images are provided as RGB tiles  
- Labels are binary segmentation masks indicating tree canopy vs. non-canopy  
- The dataset covers mixed rural/urban regions with varied vegetation density  
- Training set includes paired image + mask tiles  
- Evaluation metrics typically include **IoU**, **F1-score**, and **pixel accuracy**  

The dataset is not included in this repository due to competition rules and licensing restrictions.

## About This Project
This project adapts the original **Swin-Unet** model (designed for medical image segmentation) to a remote-sensing task:

- Custom dataset loader for Solafune tiles  
- Custom training/validation splits  
- Adjusted hyperparameters for canopy detection  
- Modified configuration files  
- Added evaluation metrics specific to segmentation  
- Experiment tracking and visualization (loss curves, IoU curves, etc.)

## Reference to Original Work
This repository is based on:

**Swin-Unet**  
https://github.com/HuCaoFighting/Swin-Unet

The architecture was proposed in:

```bibtex
@InProceedings{swinunet,
author    = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
title     = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
booktitle = {Proceedings of the European Conference on Computer Vision Workshops (ECCVW)},
year      = {2022}
}
