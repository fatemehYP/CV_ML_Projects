import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
    Implementing a Multiobject Tracker class from OpenCV which has a very basic implementation.
    It processes the tracked objects independently without any optimization across the tracked objects.
'''

def choose_tracker(tracker_type):
    if tracker_type == "Boosting":
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "TLD":
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == "MEDIANFLOW":
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('available trackers are:')
        for name in tracker_type:
            print(name)
    return tracker


def initial_bbox():
    global color
    color = []
    for i in range(3):
        color.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    bboxes = [(471, 250, 66, 159), (349, 232, 69, 102)]
    return bboxes


def draw_bbox(frame, boxes):
    for i, box in enumerate(boxes):
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, pt1, pt2, color[i], 4, cv2.LINE_AA)
        result.write(frame)


def track():
    while True:
        success, frame = video.read()
        if not success:
            break
        ok, boxes = multitracker.update(frame)
        draw_bbox(frame, boxes)


if __name__ == "__main__":

    tracker_type = ['Boosting', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT', 'MOSSE']
    tracker_type = tracker_type[6]

    video = cv2.VideoCapture('videos/cycle.mp4')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    result = cv2.VideoWriter('results/output_multiple_tracker.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

    success, frame = video.read()
    if not success:
        print('Cannot read video file')

    multitracker = cv2.MultiTracker_create()

    bboxes = initial_bbox()
    for bbox in bboxes:
        multitracker.add(choose_tracker(tracker_type), frame, bbox)

    track()
