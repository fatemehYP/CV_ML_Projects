import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

'''
    Using a HOG Descriptor + SVM classifier for Object Detection.
    Training a People Detector using INRIA’s Pedestrian dataset.
    Dataset contains two sub-folders:
        train_64x128_H96 - contains the cropped images of pedestrians and resized to 64x128 ( Positive Examples )
        test_64x128_H96 - contains cropped images which do not contain pedestrians. ( Negative Examples )
    The training data can also be downloaded from this link: https://www.filepicker.io/api/file/VQKdmzKqSLWruVhx7Xdd
    At the end we compare our model with OpenCV’s default People Detector.
    (Green rectangles for results from our model and red boxes for results from OpenCV’s people detector.)
'''


def svmInit(c, gamma):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(c)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setTermCriteria((cv2.TERM_CRITERIA_EPS +
                           cv2.TERM_CRITERIA_MAX_ITER,
                           1000, 1e-3))
    return model


def svmTrain(model, trainData, trainLabel):
    model.train(trainData, cv2.ml.ROW_SAMPLE, trainLabel)
    return model


def svmPredict(model, testData):
    return model.predict(testData)[1].ravel()


def svmEvaluate(model, testData, testLable):
    predict = svmPredict(model, testData)
    accuracy = (predict == testLable).mean()
    print('Percentage Accuracy: %.2f %%' % (accuracy * 100))
    return accuracy


def prepareData(data):
    feature_length = len(data[0])
    features = np.float32(data).reshape(-1, feature_length)
    return features


def computeHOG(hog, data):
    hogData = []
    for image in data:
        hogFeatures = hog.compute(image)
        hogData.append((hogFeatures))
    return hogData


def getlabel(path, label):
    image_label = []
    images = [cv2.imread(file) for file in glob.glob(path + '/*.jpg')]
    for count, image in enumerate(images):
        image_label.append(label)
    return images, image_label


if __name__ == '__main__':

    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = True
    nlevels = 64
    signedGradient = False

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    cwd = os.getcwd()
    train_neg_path = cwd + '/images/INRIAPerson/train_64x128_H96/negPatches'
    train_pos_path = cwd + '/images/INRIAPerson/train_64x128_H96/posPatches'
    test_neg_path = cwd + '/images/INRIAPerson/test_64x128_H96/negPatches'
    test_pos_path = cwd + '/images/INRIAPerson/test_64x128_H96/posPatches'

    train_neg, train_neg_label = getlabel(train_neg_path, -1)
    train_pos, train_pos_label = getlabel(train_pos_path, 1)
    test_neg, test_neg_label = getlabel(test_neg_path, -1)
    test_pos, test_pos_label = getlabel(test_pos_path, 1)

    train_images = np.concatenate((np.array(train_pos), np.array(train_neg)), axis=0)
    train_labels = np.concatenate((np.array(train_pos_label), np.array(train_neg_label)), axis=0)
    test_images = np.concatenate((np.array(test_pos), np.array(test_neg)), axis=0)
    test_labels = np.concatenate((np.array(test_pos_label), np.array(test_neg_label)), axis=0)

    train_hog = computeHOG(hog, train_images)
    test_hog = computeHOG(hog, test_images)

    train_features = prepareData(train_hog)
    test_features = prepareData(test_hog)

    model = svmInit(c=0.01, gamma=0.0)
    model = svmTrain(model, train_features, train_labels)
    model.save('results/pedestrian.yml')

    saved_model = cv2.ml.SVM_load('results/pedestrian.yml')
    accuracy = svmEvaluate(saved_model, test_features, test_labels)

    sv = model.getSupportVectors()
    rho, aplha, svidx = model.getDecisionFunction(0)
    svmDetector = np.zeros(sv.shape[1] + 1, dtype=sv.dtype)
    svmDetector[:-1] = -sv[:]
    svmDetector[-1] = rho

    # set our SVMDetector in HOG
    hog.setSVMDetector(svmDetector)

    # opencv default pedestrian detection

    hogDefault = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                   histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    svmDetectorDefault = cv2.HOGDescriptor_getDefaultPeopleDetector()
    hogDefault.setSVMDetector(svmDetectorDefault)

    # read test image
    filename = cwd + "/images/pedestrians/1.jpg"
    queryImage = cv2.imread(filename, cv2.IMREAD_COLOR)

    # We will run pedestrian detector at an fixed height image
    finalHeight = 800.0
    scale = finalHeight / queryImage.shape[0]
    queryImage = cv2.resize(queryImage, None, fx=scale, fy=scale)

    # detectMultiScale will detect at nlevels of image by scaling up
    # and scaling down resized image by scale of 1.05
    bboxes, weights = hog.detectMultiScale(queryImage, winStride=(8, 8),
                                           padding=(32, 32), scale=1.05,
                                           finalThreshold=2, hitThreshold=1.0)

    bboxes2, weights2 = hogDefault.detectMultiScale(queryImage, winStride=(8, 8),
                                                    padding=(32, 32), scale=1.05,
                                                    finalThreshold=2, hitThreshold=0)

    # draw detected bounding boxes over image
    for bbox in bboxes:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(queryImage, (x1, y1), (x2, y2),
                      (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    for bbox in bboxes2:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(queryImage, (x1, y1), (x2, y2),
                      (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

    plt.imshow(queryImage[:, :, ::-1])
    plt.show()
