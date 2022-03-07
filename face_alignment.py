import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
import math

''' Face Alignment using the 5-point model.'''


# In the 5-point model, the landmark points consists of 2 points at the corners of the eye
#   for each eye and one point on the nose-tip


def face_Detector(image):
    faceDetector = dlib.get_frontal_face_detector()
    faceBboxes = faceDetector(image, 0)
    print("Number of faces detected: ", len(faceBboxes))
    return faceBboxes


def land_marks(image, faceBboxes):
    predictorPath = 'models/shape_predictor_5_face_landmarks.dat'
    landMarkDetector = dlib.shape_predictor(predictorPath)
    for i in range(len(faceBboxes)):
        bbox = dlib.rectangle(int(faceBboxes[i].left()), int(faceBboxes[i].top()), int(faceBboxes[i].right()),
                              int(faceBboxes[i].bottom()))
        landMarks = landMarkDetector(image, bbox)
        points = []
        for p in landMarks.parts():
            point = (p.x, p.y)
            points.append(point)
        points = np.array(points)
    return points


def similarity_transform(eyecornerSrc, eyecornerDst):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    inPts = np.copy(eyecornerSrc).tolist()
    outPts = np.copy(eyecornerDst).tolist()

    # The third point is calculated so that the three points make an equilateral triangle
    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]

    inPts.append([int(xin), int(yin)])

    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]

    outPts.append([int(xout), int(yout)])

    # Now we can use estimateRigidTransform for calculating the similarity transform.
    transform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
    return transform[0]


def alignment(size, image, points):
    h, w = size
    if len(points) == 68:
        eyecornerSrc = [points[36], points[45]]
    elif len(points) == 5:
        eyecornerSrc = [points[2], points[0]]
    # Corners of the eye in normalized image
    eyecornerDst = [(int(0.3 * w), int(h / 3)),
                    (int(0.7 * w), int(h / 3))]

    transform = similarity_transform(eyecornerSrc, eyecornerDst)
    result = np.zeros(image.shape, dtype=image.dtype)
    result = cv2.warpAffine(image, transform, (w, h))
    points2 = np.reshape(points,
                         (points.shape[0], 1, points.shape[1]))

    pointsOut = cv2.transform(points2, transform)

    pointsOut = np.reshape(pointsOut,
                           (points.shape[0], points.shape[1]))
    return result, pointsOut


if __name__ == "__main__":
    image = cv2.imread("images/face1.png")
    faceBboxes = face_Detector(image)
    points = land_marks(image, faceBboxes)
    h = 600
    w = 600
    im = np.float32(image) / 255.0
    res, points = alignment((h, w), im, points)
    res = np.uint8(res * 255)
    plt.figure(figsize=(20, 8))
    plt.imshow(res[:, :, ::-1])
    plt.title("Aligned Image")
    plt.show()
