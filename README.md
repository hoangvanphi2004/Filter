# Filter App
## Overview
This app is an app to apply hat filter to a face by predict keypoints and use some math to find the hat position. </br> In this app, I use resnet34 model to predict the keypoints. I also try to use vgg11 to predict points but its not as good as resnet34.</br></br> Some of the frameworks/libraries i use:
- pytorch
- pytorch-lightning
- albumentations
- opencv
## Feature
- Show facial keypoints
- Apply hat filter
## How to use
- First you must have python installed in for computer. You can go to [this page](https://www.python.org/downloads/) to download the latest version 
- Download/clone this git repo
- Open file Install.bat
- After that you can use the app by open commandline in the folder you download this repo into
  1. You can show the predict keypoints by using
     ```
     python index.py
     ```
     or
     ```
     python index.py keypoints
     ```
  2. You can apply filter by using
     ```
     python index.py filter path/to/your/file  
     ```
     You can change the hat poition relative to your face in vertical direction by using
     ```
     python index.py filter path/to/your/file distanceRelativeToInitialPosition
     ```
     Or you can just apply the default filter (LuffyHat.png)
     ```
     python index.py filter
     ```
