# CLEVR-ER - A relational synthetic dataset with liquids

The focus of this dataset is to enable diagnosis of relations understanding. This project creates synthetic data similar to CLEVR, only it supports later versions of Blender (2.93 <= v <= 3.0.0) and allows liquid properties for the objects created for relations predictions. It also suppurt a few other sort of relation as comperative relations and spatial relations. 

# Download 

[Here](https://drive.google.com/file/d/1thvwm6BochjJcTgCNSlOvbULzlSZl4q6/view?usp=sharing) you can download the dataset we used for the relations benchmark. There are 5000 samples in this link. you can render more samples with the code if you want.

# Examples 

<table>
  <tr>
    <td> <img src="https://github.com/yoterel/CLEVR-ER/blob/main/resource/splash1.png"  alt="1" width = 256px height = 256px ></td>
    <td> <img src="https://github.com/yoterel/CLEVR-ER/blob/main/resource/splash2.png"  alt="2" width = 256px height = 256px ></td>
    <td> <img src="https://github.com/yoterel/CLEVR-ER/blob/main/resource/splash3.png"  alt="3" width = 256px height = 256px ></td>
   </tr>
</table>


# Baseline

Relations\model| random | vgg-features | Clip ViT | Clip RN50 | vgg-no-location-input
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 
