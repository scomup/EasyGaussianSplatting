# Simple Gaussian Splatting

## What is this? 

For learning purposes, we offer a set of tools and documentation to study 3D Gaussian Splatting. We also plan to provide an unofficial implementation of the paper [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).


## Overview 

* Detailed documentation to demonstrate the mathematical principles of 3D Gaussian Splatting
    - [x] [Documentation for forward (render image)](docs/forward.pdf)
    - [x] [Documentation for backward (training)](docs/backward.pdf)

- Based on our documentation, re-implement 3D Gaussian Splatting.
    - [x] Forward on CPU
    - [x] Forward on GPU
    - [ ] Backward on CPU
    - [ ] Backward on GPU

- Provide tools for learning 3D Gaussian Splatting.
    - [x] A simple viewer based on pyqtgraph for showing 3D Gaussian data (trained model).
    - [x] A demo showing how spherical harmonics work.

## Requirements 

```bash
pip3 install -r requirements.txt
pip install pygauspilt/.
```

## Forward process (render image)

Given camera information, render 3D Gaussian data onto the 2D image using Python.

CPU version.
```bash
python3 forword.py --ply='THE_PATH_OF_YOUR_TRAINED_PLY_FILE'
```

GPU version.
```bash
python3 forword_gpu.py --ply='THE_PATH_OF_YOUR_TRAINED_PLY_FILE'
```
![forword demo](imgs/forword.png)

## 3D Gaussian Viewer 

A fast 3D Gaussian splatting viewer for showing 3D Gaussian data. 

```bash
python3 gaussian_viewer.py --ply='THE_PATH_OF_YOUR_TRAINED_PLY_FILE'
```

<img src="imgs/viewer.gif" width="640px">



## Spherical harmonics demo

A demo showing how spherical harmonics work.

```bash
python3 sh_demo.py
```

![sh demo](imgs/sh_demo.gif)
<span style="font-size: 80%; color: Gray;">"The ground truth Earth image is modified from [URL](https://commons.wikimedia.org/wiki/File:Solarsystemscope_texture_8k_earth_daymap.jpg). By Solar System Scope. Licensed under CC-BY-4.0"</span>