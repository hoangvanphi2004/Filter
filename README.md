<div align="center">

  # Filter App
  ![image](https://github.com/user-attachments/assets/0cdc3ccf-2bc6-47c4-90a7-f5cabf339642)

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
First you must have python installed in for computer. You can go to [this page](https://www.python.org/downloads/) to download the latest version 

![image](https://github.com/user-attachments/assets/cdd7de4c-d0ca-4d91-ba0c-d40d867251f8)

After that, clone this git repo to the folder you want. Depend on you OS, you will need to run different file:
- If you are using Linux-like OS, you must run InstallLinux.sh
- If you are using Window, you must run InstallWindow.bat

Before run any filter, you need to active virtual environment. To active the virtual environment, also depend on your OS, you will need to run:
- With Window:

```bash
./env/Scripts/activate.bat
```

- With Linux-like OS:

```bash
source ./env/bin/activate
```

### Hat Filter
To run hat filter, the most basic commad you can run is:

```bash
python3 index.py hat-filter
```

This command will use the LuffyHat.png as the filter image to apply the filter

To use other image, use can add the specific path to your filter image:

```bash
python3 index.py hat-filter [path_to_your_iamge]
```

You can also change the hat position in vertical axis, to do it you can add the "distance from the face" value:

```bash
python3 index.py hat-filter [path_to_your_iamge] [distance]
```

If the distance is short, the hat will be near your face. If the distance is long, the hat will be far from your face.
![Untitled](https://github.com/user-attachments/assets/f6ec7ade-e8e6-4469-bbd5-c29ba5f7a49e)

### Mask Filter
To run mask filter, you need to have a mask file. The mask need to have white or black background (no background is even better). Some time the mask may be broken because the edge is blured. In that case, you need to remove that blur. To do this, you need to run this command:

```bash
python3 adjustBackground.py [pixel-range] [path_to_your_mask]
```

Where the pixel-range is the range of color in RGB you want to remove in background. If this range is too small, the background can not unblur, but if this range is too big, the background might invade the edge make the whole image become background. The recommended value is 30, but you can play around with it to feel the color ;)

After we run the above command, a window will appear like this

![image](https://github.com/user-attachments/assets/830b0e85-890f-4895-b965-f9eae68a6d75)

Double click on the area that you want to be the background, the background area would turn gray.

![image](https://github.com/user-attachments/assets/d53dc0b3-d317-4ec9-ae6e-514cf53eff8d)


After you finish, press 'q' to save the mask file. The file would automatically save as 'mask.png'.

Some example of bad pixel-range value:
- The range is too small. As you can see the border of the image have many tiny white dot that we dont expect. It will make the mask look ugly
  
  ![image](https://github.com/user-attachments/assets/65eb620a-d22d-4b26-abf1-74725ac4c2bc)
  
- The range is too big. The range is too big that the background start invade the mask area, make it look ugly too

  ![image](https://github.com/user-attachments/assets/df88d0f1-cb64-43b9-99a7-0d623ddbed23)
 
After remove the background, you need to identify the face points in the mask. The face points in the mask is corresponding to the face points in your face when you use the app. You can draw the points follow the face below, the point you set on the mask would be the correspoinding point on the face.   

![image](https://github.com/user-attachments/assets/04bba697-70ca-4b35-9b25-271f8489688d)   ![image](https://github.com/user-attachments/assets/db03c009-02f6-4e31-ab62-2a7fe9c644ea)

To assign the points to the mask, you need to run this command:

```bash
python3 keypoints_predict.py
```

A window will appear, it will contain the mask like previous step. double click on the point you want to assign. Remember, you need to assign point-after-point respectively. The number of the point  you are assigning is appeared in the terminal. If you mis-assign a point, you can always re-assign it by press 'd'. If you want to assign all the point from beginnning, you can press 'r'. After you assigned all 68 points, the window will auto close. If you dont want to assign anymore, press 'q'.

![image](https://github.com/user-attachments/assets/203e9894-cac9-4869-86f3-c04224f0e6c6)

Now you have all the requirement. Let just run the filter!

Run this command to apply the mask:

```bash
python3 index.py mask-filter
```

### Show Keypoints 
In the previous section, you can apply the points in the mask to the corresponding points in your face. But how can you know the exact points? If you want to know, you can run the command below to show the points in your face:

```bash
python3 index.py 
```

Or 

```bash
python3 index.py keypoints
```
### Other features
If you want to make any of the filter above floating, you can just add "floating-mask" to the end of the command, for example:

```bash
python3 index.py mask-filter floating-mask
```

Or if you want to save the video, you can always do that by adding the term "save-video" to the end:

```bash
python3 index.py mask-filter save-video
```

### Train model
Beside the filtering features, i also provide a code to train the model to predict the keypoint. To do that you need to download <a href="https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/">this dataset</a> and put it in the project folder. After that you can run:

```bash
python3 train.py [path_to_save_file]
```
You need to specificly indentify the name of the file in the path. It would require you to have gpu in your computer. After you finished training your model, you can open file keypoints_predict.py and change the "resnet-for-face-points-recognize-state-dict.pth" to the path to your model to use your model.

![image](https://github.com/user-attachments/assets/aaec1f38-9107-462a-bf94-a996556948d0)

## Image About The App
![Filter_screenshot2_30 08 2024](https://github.com/user-attachments/assets/f3d54b72-e641-415b-8330-7ffb82abb465)
![Filter_screenshot1_30 08 2024](https://github.com/user-attachments/assets/afa2dac8-8f59-45ae-a9a5-13206ba4c0e1)
![Filter_screenshot_30 08 2024](https://github.com/user-attachments/assets/84b844f3-0667-4379-8593-39a3930d5d43)

## Credit
- Many thank to <a href="https://www.youtube.com/watch?v=dK-KxuPi768">this</a> video series for teaching me alot about opencv and delaunay triangulatiton 
- Thank to <a href="https://abel.math.harvard.edu/archive/116_fall_03/handouts/kalman.pdf">this</a> paper which help me alot in understanding Kalman Filter 
- Thank to Nguyen Dang Huynh and Le Vu Minh for helping me in the project
- And also, to the person who bring me here, the one I respect the most, <a href="https://github.com/PAD2003"> Phan Anh Duc </a>

## Contact
If you have any idea or find any mistake in the project, please tell me via:
- Email: hoangvanphi2004@gmail.com
- Facebook: fb.com/haongvanphi2004
<div align="center">

  # ~~~ Thanks for visiting my repo ~~~
  
</div>
