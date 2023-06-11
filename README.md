# CLEVR-ER - A relational synthetic dataset with liquids

CLEVR-ER is a dataset for diagnosis of relations understanding. It creates synthetic data similar to CLEVR but it adds relational information between the objects.
It suppurts 6 sorts of relation as comperative relations and spatial relations as well as action and liquid-based relations (see the table bellow for the exact relations).  
It supports advanced versions of Blender (2.93 <= v <= 3.0.0). 
For more details, see our [G-Slides](https://docs.google.com/presentation/d/18cVUpvwEqYvGGdlF3Xw5WhNWMEfh4aGvlJjgJAnGhG0/edit?usp=sharing)

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

# Running Example
To create random data, you can run the following. It is recommended to follow the instructions of the original CLEVR dataset to have full flexibility in those configurations.
```
/Applications/Blender.app/Contents/MacOS/blender  --background --python render_images.py -- --num_images 1 --min_objects 2 --max_objects 2 --liquid_simulation
```

For training the benchmark, you can simply run the model.py file. For help and configuration details add the flag -h.

# Installation
Follow the exact installation instructions of te original CLEVR but use Blender version of (2.93 <= v <= 3.0.0) to allow liquid properties.

# Baseline Results

| Relations\model| random | vgg-features | Clip ViT | Clip RN50 | vgg no location input
| ---            |    --- | ---          | ---      |---        |---
| Greater        | 0.33    | 0.87        | 0.47     | 0.48      | 0.858 
| Higher         | 0.5     | 1.00        | 1.00     | 1.00      | 0.996 
| Sparklier      | 0.50    | 0.874       | 0.52     | 0.49      | 0.89  
| RelativeLocation|0.25    | 0.975       | 0.98     | 0.98      | 0.84 
| Liquid         | 0.2     | 0.993       | 0.96     | 0.95      | 0.994 
| Closer than    | 0.5     | 0.88        | 0.82     | 0.80      | 0.84 
| #Average       | 0.38    | 0.932       | 0.784     | 0.783    | 0.903 
