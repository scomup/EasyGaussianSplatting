# Simple gaussian splatting

## What is this? 

For learning purposes, we offer a set of tools and documentation to study 3D Gaussian Splatting. we also plan to provide an unofficial implementation of paper [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).


## Overview 

* Detailed documentation to demonstrate the mathematical principles of 3D Gaussian Splatting
    - [x] [Documentation for forward (render iamge)](docs/forward.pdf)
    - [ ] Documentation for backword (training)

- Based on our the documentation, re-implement 3D Gaussian Splatting.
    - [x] forward process
    - [ ] backword process

- Provide tools for learning 3D Gaussian Splatting.
    - [x] a simple viewer based on pyqtgraph for showing 3D Gaussian data (trained model). 

## Requirements 

```bash
pip3 install -r requirements.txt
pip install simple_gaussian_reasterization/.
```

## Forward process (render image)

Given camera information, render 3D Gaussian data onto the 2d image by python.

```bash
python3 forword --ply='THE_PATH_OF_YOUR_TRAINED_PLY_FILE'
```
![forword demo](imgs/forword.png)

## Viewer 

A 3D Gaussian splatting viewer for showing 3D Gaussian data. 

```bash
python3 viewer/viewer --ply='THE_PATH_OF_YOUR_TRAINED_PLY_FILE'
```
![viewer demo](imgs/viewer.gif)

