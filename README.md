# Computer Vision and Machine Learning Projects
This folder contains different AI applications. List of these applications can be seen in the table of contents.
## Table of Contents
* [facial_landmark_detection.py](#facial_landmark_detection)
* [face_smile_detection.py](#face_smile_detection)
* [optical_flow_tracker.py](#optical_flow_tracker)
* [pedestrian_detection.py](#pedestrian_detection)
* [delaunay_voronoi.py](#delaunay_voronoi)
* [triangle_warping.py](#triangle_warping)
* [face_averaging.py](#face_averaging)

[comment]: <> (* [Screenshots]&#40;#screenshots&#41;)

[comment]: <> (* [Setup]&#40;#setup&#41;)

[comment]: <> (* [Usage]&#40;#usage&#41;)

[comment]: <> (* [Project Status]&#40;#project-status&#41;)

[comment]: <> (* [Room for Improvement]&#40;#room-for-improvement&#41;)

[comment]: <> (* [Acknowledgements]&#40;#acknowledgements&#41;)

[comment]: <> (* [Contact]&#40;#contact&#41;)
<!-- * [License](#license) -->

## facial_landmark_detection

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

## face_smile_detection
Object Detection using Haar Cascades.

This script uses haar cascade for detecting face and smile in an image. The trained model for face detection and smile detection is downloaded from this link:

[https://github.com/opencv/opencv/tree/master/data/haarcascades]

The main function used in the script is **detectMultiscale**.

Result:
![Example screenshot](results/face_smile_detection.png)

## optical_flow_tracker
Motion estimation using optical flow.
- Step1: **Detect Corners for tracking them**
  
  Using the Shi Tomasi corner detection algorithm to find some points which will be tracked over the video.
  
  It is implemented in OpenCV using the function **goodFeaturesToTrack**.

- Step2: **Set up the Lucas Kanade Tracker**
  
  After detecting certain points in the first frame, they will be tracked in the next frame.
  
  This is done using Lucas Kanade algorithm. 

Result:

![Example screenshot](results/optical_flow_tracker.png)

## pedestrian_detection
Using a HOG Descriptor + SVM classifier for Object Detection.

Using INRIA’s Pedestrian dataset for training a people detector. Dataset contains two sub-folders:
- train_64x128_H96 - contains the cropped images of pedestrians and resized to 64x128 ( Positive Examples )
- test_64x128_H96 - contains cropped images which do not contain pedestrians. ( Negative Examples )
  
The training data can also be downloaded from this link:

[https://www.filepicker.io/api/file/VQKdmzKqSLWruVhx7Xdd]

Results: At the end we compare our model with OpenCV’s default People Detector.
    (Green rectangles for results from our model and red boxes for results from OpenCV’s people detector.)

![Example screenshot](results/pedestrian_detection.png)

## delaunay_voronoi

![Example screenshot](results/delaunay_voronoi.png)

## triangle_warping

![Example screenshot](results/triangle_warping.png)

## face_averaging

**Creating an average face (combination of faces) using OpenCV.**

The steps for generating an average face given a set of facial images is described below:

- **Step 1 : Facial Feature Detection**

  For each facial image we calculate 68 facial landmarks using Dlib.
  (details of landmark detection is described in **facial_landmark_detection.py**)
- **Step 2 : Coordinate Transformation**

  The input facial images can be of very different sizes. So we need a way to normalize
  the faces and bring them to the same reference frame. We can use **estimateAffinePartial2D** function
  to find the similarity transform and convert the input image of size
  (m,n) to output image coordinates of size (m'×n').
  Once a similarity transform is calculated, it can be used to transform the input image and 
  the landmarks to the output coordinates. The image is transformed using **warpAffine** and 
  the points are transformed using the transform function.
  (We chose the corners of the eyes to be at ( 0.3 x width, height / 3 ) and ( 0.7 x width , height / 3 ).
  OpenCV requires you to supply at least 3 point pairs. We can simply hallucinate a third point such that it forms
  an equilateral triangle with the two known points).
- **Step 3 : Face Alignment**
  
  Next, we use a trick to align all the facial features. We will use 68 points to divide the images into
  triangular regions and align these regions before averaging pixel values.
  
  - **Calculate Mean Face Points**
  
    To calculate the average face where the features are aligned, we first need to calculate the average of all
    transformed landmarks in the output image coordinates. This is done by simply averaging the x and y values of
    the landmarks in the output image coordinates.
  - **Calculate Delaunay Triangulation**
    
    We can use these 68 points, and 8 points on the boundary of the output image to calculate a Delaunay Triangulation.
    (to calculate a Delaunay Triangulation refer to **delaunay_voronoi.py**)
    
  - **Warp Triangles**
  
    The collection of triangles in the input image is warped to the corresponding triangle in the output image.
    (for details about warp triangles refer to **triangle_warping.py**)
- **Step 4 : Face Averaging**

  To calculate the average image, we can simply add the pixel intensities of all warped images and divide by the
  number of images.

Results:

![Example screenshot](results/face_averaging.png)