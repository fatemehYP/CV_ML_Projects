# Computer Vision Projects
This folder contains different AI applications. List of these applications can be seen in the table of contents.
## Table of Contents
* [optical_flow_tracker.py](#optical_flow_tracker.py)
* [facial_landmark_detection.py](#facial_landmark_detection.py)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->

## optical_flow_tracker.py
Motion estimation using optical flow.
- Step1: Detect Corners for tracking them
  
  Using the Shi Tomasi corner detection algorithm to find some points which will be tracked over the video.
  
  It is implemented in OpenCV using the function goodFeaturesToTrack.

- Step2: Set up the Lucas Kanade Tracker
  
  After detecting certain points in the first frame, they will be tracked in the next frame.
  
  This is done using Lucas Kanade algorithm. 

Result:

![Example screenshot](results/res12.png)

## facial_landmark_detection.py

Landmark detection is a two step process:

- Step1. **Face Detection**:
  
  For best results we should use the same face detector used in training the landmark detector.
  
  Dlib has a built-in face detector which can be accessed using **get_frontal_face_detector()**
  
  (Dlib’s face detector is based on Histogram of Oriented Gradients features and Support Vector Machines (SVM))

- Step2. **Landmark detection**
  
  The landmark detector finds the landmark points inside the face rectangle.
  
  The **shape_predictor** class implements Dlib’s facial landmark detector.

  Dlib’s landmark detector needs two inputs:
  - Input image.
  - Face rectangle

  The output landmarks are stored in an instance of the class full_object_detection.
  
  A few important methods of full_object_detection are listed below
  - **num_parts()** : Number of landmark points.
  - **part(i)** : The ith landmark point
  - **part(i).x()** and **part(i).y()** can be used to access the x and y coordinates of the ith landmark point.

Result:
This script detects facial landmarks and plot them on a face image.

![Example screenshot](results/Facial%20Landmark%20detector.jpg)



