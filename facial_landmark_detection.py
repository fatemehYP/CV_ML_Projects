from cv2 import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt

''' Facial Landmark Detection using Dlib '''


# We will show how to detect facial landmarks and plot them on a face image.
# Landmark detection is a two step process:
# 1. Face Detection (For best results we should use the same face detector used in training the landmark detector)
#    Dlib’s face detector is based on Histogram of Oriented Gradients features and Support Vector Machines (SVM)
# 2. Landmark detection

def face_Detector(image):
    faceDetector = dlib.get_frontal_face_detector()
    faceBboxes = faceDetector(image, 0)
    print("Number of faces detected: ", len(faceBboxes))
    return faceBboxes


def land_marks(image, faceBboxes):
    predictorPath = 'models/shape_predictor_68_face_landmarks.dat'
    landMarkDetector = dlib.shape_predictor(predictorPath)
    for i in range(len(faceBboxes)):
        bbox = dlib.rectangle(int(faceBboxes[i].left()), int(faceBboxes[i].top()), int(faceBboxes[i].right()),
                              int(faceBboxes[i].bottom()))
        landMarks = landMarkDetector(image, bbox)
        draw_face(image, landMarks)
        # draw_face2(image, landMarks)


def draw_polyline(image, landmarks, start, end, isClosed=False):
    points = []
    for i in range(start, end + 1):
        point = (landmarks.part(i).x, landmarks.part(i).y)
        points.append(point)
    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed, (255, 200, 0), 3, cv2.LINE_8)


# Use this function for 68-points facial landmark detector model
def draw_face(image, landmarks):
    assert (landmarks.num_parts == 68)
    draw_polyline(image, landmarks, 0, 16)  # Jaw line
    draw_polyline(image, landmarks, 17, 21)  # Left eyebrow
    draw_polyline(image, landmarks, 22, 26)  # Right eyebrow
    draw_polyline(image, landmarks, 27, 30)  # Nose bridge
    draw_polyline(image, landmarks, 30, 35, True)  # Lower nose
    draw_polyline(image, landmarks, 36, 41, True)  # Left eye
    draw_polyline(image, landmarks, 42, 47, True)  # Right Eye
    draw_polyline(image, landmarks, 48, 59, True)  # Outer lip
    draw_polyline(image, landmarks, 60, 67, True)  # Inner lip

# Use this function for any model other than 68 points facial_landmark detector model
def draw_face2(image, landmarks, color=(0, 255, 0), radius=3):
    for p in landmarks.parts():
        cv2.circle(image, (p.x, p.y), radius, color, -1)


if __name__ == "__main__":
    image = cv2.imread("images/family.jpg")
    faceBboxes = face_Detector(image)
    land_marks(image, faceBboxes)
    plt.figure(figsize=(20, 8))
    plt.imshow(image[:, :, ::-1])
    plt.title("Facial Landmark detector")
    plt.show()
