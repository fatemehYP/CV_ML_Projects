from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

def warping(inputImage, outputImage, inputTriangle, outputTriangle):
    rec1 = cv2.boundingRect(inputTriangle)
    rec2 = cv2.boundingRect(outputTriangle)
    inputImage_cropped = inputImage[rec1[1]:rec1[1] + rec1[3], rec1[0]:rec1[0] + rec1[2]]
    tri1Cropped = []
    tri2Cropped = []
    for i in range(0, 3):
        tri1Cropped.append(((inputTriangle[0][i][0] - rec1[0]), (inputTriangle[0][i][1] - rec1[1])))
        tri2Cropped.append(((outputTriangle[0][i][0] - rec2[0]), (outputTriangle[0][i][1] - rec2[1])))

    warptransform = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))
    outputImage_cropped = cv2.warpAffine(inputImage_cropped, warptransform, (rec2[2], rec2[3]),
                                         None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((rec2[3], rec2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1, 1, 1), 16, 0)
    outputImage_cropped = outputImage_cropped * mask

    outputImage[rec2[1]:rec2[1] + rec2[3], rec2[0]:rec2[0] + rec2[2]] = \
        outputImage[rec2[1]:rec2[1] + rec2[3], rec2[0]:rec2[0] + rec2[2]] * ((1.0, 1.0, 1.0) - mask)

    outputImage[rec2[1]:rec2[1] + rec2[3], rec2[0]:rec2[0] + rec2[2]] = \
        outputImage[rec2[1]:rec2[1] + rec2[3], rec2[0]:rec2[0] + rec2[2]] + outputImage_cropped

    color = (0, 255, 0)
    cv2.polylines(inputImage, inputTriangle.astype(int), True, color, 2, cv2.LINE_AA)
    cv2.polylines(outputImage, outputTriangle.astype(int), True, color, 2, cv2.LINE_AA)
    combined = cv2.hconcat([inputImage, outputImage])
    plt.imshow(combined[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    inputImage = cv2.imread("images/kingfisher.jpg")
    outputImage = 255 * np.ones(inputImage.shape, dtype=inputImage.dtype)
    plt.imshow(outputImage)
    inputTriangle = np.float32([[[360, 50], [60, 100], [300, 400]]])
    outputTriangle = np.float32([[[400, 200], [160, 270], [400, 400]]])
    warping(inputImage, outputImage, inputTriangle, outputTriangle)
