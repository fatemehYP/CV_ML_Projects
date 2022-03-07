from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
    Using haar cascade based face and smile detectors in OpenCV.
    Using both the cascades for detecting face and smile in an image.
    The trained model for face detection and smile detection is downloaded from this link:
    https://github.com/opencv/opencv/tree/master/data/haarcascades
'''

if __name__ == "__main__":
    test_image = cv2.imread("images/hillary_clinton.jpg")
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    smileCascade = cv2.CascadeClassifier('models/haarcascade_smile.xml')

    faceNeighborsMax = 90
    neighborStep = 10
    fig = plt.figure(figsize=(20, 8))
    count = 1

    faces = faceCascade.detectMultiScale(test_image_gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(test_image, (x, y), ((x + w), (y + h)), (255, 0, 0), 3, cv2.LINE_AA)
        faceRoiGray = test_image_gray[y: y + h, x: x + w]

    for neigh in range(1, faceNeighborsMax, neighborStep):
        smiles = smileCascade.detectMultiScale(faceRoiGray, 1.5, neigh)
        frameClone = np.copy(test_image)
        faceRoiClone = frameClone[y: y + h, x: x + w]
        for (xx, yy, ww, hh) in smiles:
            cv2.rectangle(faceRoiClone, (xx, yy), ((xx + ww), (yy + hh)), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frameClone, "number={}".format(neigh), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        fig.add_subplot(3, 3, count)
        count = count + 1
        plt.imshow(frameClone[:, :, ::-1])
    plt.show()
