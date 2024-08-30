<div align="center">

  # Filter App
  
</div>

## Overview
Deep learning have developed rapidly in the past few years, many aspect of it have been explored to increase the life quality. In the entertainment industry, many application like tiktok, facebook and so on have applied this advanced technology in their application, create thousand of trendings. And with the desire to master this technology, this repo is created to be a great step in my life long learning journey. This repo is about an filter app which you can apply a mask from an png image to your face. Beside the mask filder, i also have mask filter and a mode to make the mask floating. I have trained a model <a href="https://drive.google.com/file/d/1Cu9A3EWkNJ34Zbr6U0hkQUgG4qzMaE5P/view?usp=drive_link">here</a> so you can just download it and paste it to the repo folder. If you want to use yourown model, you can train it by yourself! i have an file to help you in this.

## Feature
I have build many features for the app, some of it can be listed here:
- Show facial keypoints
- Apply hat filter
- Apply mask filter
- Make the filter floating
To see more about the features, please read the How to use section.
## How to use
### Installation
- First you must have python installed in for computer. You can go to [this page](https://www.python.org/downloads/) to download the latest version 
- After that, clone this git repo to the folder you want.
- Depend on you OS, you will need to run different file:
  - If you are using Linux-like OS, you must run InstallLinux.sh
  - If you are using Window, you must run InstallWindow.bat
### Hat Filter 
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
