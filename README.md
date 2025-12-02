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
- Labels are binary segmentation masks indicating group_of_trees vs. individual_tree  
- The dataset covers mixed rural/urban regions with varied vegetation density  
- Training set includes paired image + mask tiles  
- Evaluation metrics typically include **Mean Pixel Accuracy (mPA)**  

The dataset is not included in this repository due to competition rules and licensing restrictions.

---

## About This Project

This project adapts the original **Swin-Unet** architecture (a hierarchical Vision Transformer with U-Net structure) 
to tree canopy segmentation. 

Modifications include:
- Custom dataset loader for Solafune tiles  
- Preprocessing and augmentation pipeline  
- Custom training/validation split (4:1)  
- Adjusted hyperparameters for remote-sensing images  
- Logging and visualization of training curves  
- Integration of **Pixel Accuracy (PA)** and **Mean Pixel Accuracy (mPA)** monitoring  
- Evaluation in a highly data-limited setting (only 150 images)

Despite stable training and strong pixel-level accuracy, transformer-based architectures underperformed in 
instance segmentation (polygon-based mAP) due to noisy boundaries and data scarcity.

---

## Evaluation Metrics

This project uses **semantic segmentation metrics**, not instance-level mAP:

### **Pixel Accuracy (PA)**
Percentage of correctly predicted pixels across the entire dataset.

### **Mean Pixel Accuracy (mPA)**
Pixel accuracy is computed **per class** (canopy vs. background), then averaged.  
mPA is more meaningful for *imbalanced datasets* like Solafune, where canopy pixels are much fewer.

During training:
- Swin-Unet reached ~0.87 PA on training and ~0.85 PA on validation :contentReference[oaicite:2]{index=2}
- Loss curves showed stable convergence over 150 epochs

---

## Project Report (Full Analysis)

A detailed technical report is included in this repository. It provides comprehensive evaluation of five deep learning 
architectures on the Solafune dataset:

Models compared:
- **YOLOv11 (Nano / Small / Medium / Large)**
- **Mask R-CNN**
- **DeepLabv3**
- **Swin-Unet**
- **DINOv2**

### Key Findings from the Report
- Dataset contains *only 150 labeled images*, making it an extreme low-data scenario :contentReference[oaicite:3]{index=3}  
- **CNN-based models** (YOLOv11, Mask R-CNN) consistently outperform Vision Transformers :contentReference[oaicite:4]{index=4}  
- **Swin-Unet** trains stably and achieves strong pixel accuracy (~0.85), but performs poorly on instance-level mAP 
  (~0.02) due to boundary fragmentation in polygonization :contentReference[oaicite:5]{index=5}  
- **DINOv2** fails to generalize to evaluation images due to insufficient domain-specific pretraining and extreme 
  data scarcity :contentReference[oaicite:6]{index=6}  
- YOLOv11-Large achieves the highest weighted mAP (~0.28) among all tested architectures :contentReference[oaicite:7]{index=7}  
- Transformer models require **far more data** or **domain-aligned pretraining** to be competitive in remote sensing

### Included in the Report
- Loss curves for all architectures  
- Pixel accuracy curves for DINOv2 and Swin-Unet (pages 5–6) :contentReference[oaicite:8]{index=8}  
- Visual segmentation comparisons across all five models (Figure on page 7) :contentReference[oaicite:9]{index=9}  
- Discussion on CNN vs. Transformer behavior under low-data constraints  
- Final conclusions on model suitability for ecological applications

---

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
