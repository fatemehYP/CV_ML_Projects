import cv2
import numpy as np

'''
    Motion estimation using optical flow.
    Step1: Detect Corners for tracking them
    Using the Shi Tomasi corner detection algorithm to find some points which will be tracked over the video.
    It is implemented in OpenCV using the function goodFeaturesToTrack.
    Step2: Set up the Lucas Kanade Tracker
    After detecting certain points in the first frame, they will be tracked in the next frame.
    This is done using Lucas Kanade algorithm.
'''


def feature_extractor():
    param_track = dict(maxCorners=numCorners, qualityLevel=0.3, minDistance=7, blockSize=7)
    points = cv2.goodFeaturesToTrack(gray_old_frame, mask=None, **param_track)
    return points


def tracker(gray_old_frame, old_point):
    param_flow = dict(winSize=(15, 15), maxLevel=2,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    mask = np.zeros_like(old_frame)
    color = np.random.randint(0, 255, (numCorners, 3))
    count = 0
    while True:
        retval, frame = video.read()
        if retval:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            count += 1
            newPts, status, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_frame, old_point, None, **param_flow)
            good_old = old_point[status == 1]
            good_new = newPts[status == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2, cv2.LINE_AA)
                cv2.circle(frame, (int(a), int(b)), 3, color[i].tolist(), -1)
            display_frame = cv2.add(frame, mask)
            output.write(display_frame)
            if count > 50:
                break
            gray_old_frame = gray_frame.copy()
            old_point = good_new.reshape(-1, 1, 2)
        else:
            break


if __name__ == "__main__":
    video = cv2.VideoCapture("videos/cycle.mp4")
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output = cv2.VideoWriter('results/sparse-output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, (width, height))

    # Take first frame and find corners in it
    retval, old_frame = video.read()
    gray_old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    numCorners = 100
    old_point = feature_extractor()
    tracker(gray_old_frame, old_point)