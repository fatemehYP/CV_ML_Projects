from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
    implement a simple Video Stabilizer using a technique called Point Feature Matching in OpenCV library.
    Video stabilization refers to a family of methods used to reduce the effect of camera motion on the final video.
    This is a fast and robust implementation of a digital video stabilization algorithm.
'''


def goodFeatures(frame):
    param_good = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    features = cv2.goodFeaturesToTrack(frame, **param_good)
    return features


def opticalFlow(pre_frame, curr_frame, pre_features):
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(pre_frame, curr_frame, pre_features, None)
    # assert pre_features.shape == nextPts.shape
    index = np.where(status == 1)[0]
    new_good = nextPts[index]
    old_good = pre_features[index]
    return new_good, old_good


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    kernel = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smooth = np.convolve(curve_pad, kernel, mode='same')
    curve_smooth = curve_smooth[radius:-radius]
    return curve_smooth


def smooth(trajectory):
    smooth_trajectory = np.copy(trajectory)
    SMOOTHING_RADIUS = 50
    for i in range(3):
        smooth_trajectory[:, i] = movingAverage(trajectory[:, i], SMOOTHING_RADIUS)
    return smooth_trajectory


def fixBoarder(frame):
    shape = frame.shape
    T = cv2.getRotationMatrix2D((shape[1] / 2, shape[0] / 2), 0, 1.07)
    frame = cv2.warpAffine(frame, T, (shape[1], shape[0]))
    return frame


def transformEstimater(gray_pre_frame):
    transform = np.zeros((n_frame - 1, 3), dtype=np.float32)
    for i in range(n_frame - 2):
        pre_feature = goodFeatures(gray_pre_frame)
        success, curr_frame = video.read()
        if not success:
            break
        gray_curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        currPts, prePts = opticalFlow(gray_pre_frame, gray_curr_frame, pre_feature)
        estimate = cv2.estimateAffinePartial2D(prePts, currPts)
        dx = estimate[0][0, 2]
        dy = estimate[0][1, 2]
        da = np.arctan2(estimate[0][1, 0], estimate[0][0, 0])
        transform[i] = [dx, dy, da]
        gray_pre_frame = gray_curr_frame
    return transform


def stabilizer():
    for i in range(n_frame - 2):
        # frame = videoReader()
        success, frame = video.read()
        if not success:
            break
        dx = smooth_transform[i, 0]
        dy = smooth_transform[i, 1]
        da = smooth_transform[i, 2]

        new_transform = np.zeros((2, 3), np.float32)
        new_transform[0, 0] = np.cos(da)
        new_transform[0, 1] = -np.sin(da)
        new_transform[1, 0] = np.sin(da)
        new_transform[1, 1] = np.cos(da)
        new_transform[0, 2] = dx
        new_transform[1, 2] = dy
        stabilized_frame = cv2.warpAffine(frame, new_transform, (width, height))
        stabilized_frame = fixBoarder(stabilized_frame)
        result = cv2.hconcat([frame, stabilized_frame])
        # if (result.shape[1] > 1920):
        #     result = cv2.resize(result, (width, height))
        output.write(result)


if __name__ == "__main__":
    video = cv2.VideoCapture("videos/video.mp4")
    n_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    output = cv2.VideoWriter("results/video_out.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (2 * width, height))
    success, pre_frame = video.read()
    gray_pre_frame = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
    transform = transformEstimater(gray_pre_frame)
    # Trajectory
    trajectory = np.cumsum(transform, axis=0)
    smooth_trajectory = smooth(trajectory)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smooth_trajectory - trajectory
    # Calculate newer transformation array
    smooth_transform = transform + difference
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    stabilizer()
    cv2.destroyAllWindows()
    output.release()
