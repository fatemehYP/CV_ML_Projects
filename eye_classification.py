from cv2 import cv2
import numpy as np
import os
import glob

'''
     Building a HOG based classifier.
     The classifier looks at an image patch around the eyes and classifies it as wearing glasses or not wearing glasses.
'''

def getTrainTest(path, label, fraction=0.2):
    trainData = []
    testData = []
    trainLabel = []
    testLabel = []
    images = [cv2.imread(file) for file in glob.glob(path + '/*.jpg')]
    nTest = len(images) * fraction

    for count, image in enumerate(images):
        if count < nTest:
            testData.append(image)
            testLabel.append(label)
        else:
            trainData.append(image)
            trainLabel.append(label)

    return trainData, trainLabel, testData, testLabel


def svmInit(c, gamma):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(c)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
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

if __name__=="__main__":

    predicted_labels = {0: 'There is no glass', 1: 'There is a glass'}
    cwd = os.getcwd()
    with_glass_address = cwd + '/images/glassesDataset/cropped_withGlasses2'
    without_glass_address = cwd + '/images/glassesDataset/cropped_withoutGlasses2'

    winSize = (96, 32)
    blockSize = (8, 8)
    blockStride = (8, 8)
    cellSize = (4, 4)
    nbins = 9
    derivAperture = 0
    winSigma = 4.0
    histogramNormType = 1
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 1
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, 1)

    pos_trainData, pos_trainLabel, pos_testData, pos_testLabel = getTrainTest(with_glass_address, 1, 0.2)
    neg_trainData, neg_trainLabel, neg_testData, neg_testLabel = getTrainTest(without_glass_address, 0, 0.2)

    test_images = np.concatenate((np.array(pos_testData), np.array(neg_testData)), axis=0)
    train_images = np.concatenate((np.array(pos_trainData), np.array(neg_trainData)), axis=0)
    test_labels = np.concatenate((np.array(pos_testLabel), np.array(neg_testLabel)), axis=0)
    train_labels = np.concatenate((np.array(pos_trainLabel), np.array(neg_trainLabel)), axis=0)

    test_hog = computeHOG(hog, test_images)
    train_hog = computeHOG(hog, train_images)

    test_features = prepareData(test_hog)
    train_features = prepareData(train_hog)

    model = svmInit(c=2.5, gamma=0.02)
    model = svmTrain(model, train_features, train_labels)
    model.save('models/eyeGlassClassifierModel.yml')

    savedmodel = cv2.ml.SVM_load('models/eyeGlassClassifierModel.yml')
    accuracy = svmEvaluate(savedmodel, test_features, test_labels)