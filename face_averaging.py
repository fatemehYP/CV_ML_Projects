from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import dlib
import math


def landmarks_detection(images):
    faceDetector = dlib.get_frontal_face_detector()
    landMarkPath = cwd + '/models/shape_predictor_68_face_landmarks.dat'
    landMarkDetector = dlib.shape_predictor(landMarkPath)
    landmarks = []
    for i, image in enumerate(images):
        faceBbox = faceDetector(image, 0)
        bbox = dlib.rectangle(int(faceBbox[0].left()), int(faceBbox[0].top()), int(faceBbox[0].right()),
                              int(faceBbox[0].bottom()))
        landmarks.append(landMarkDetector(image, bbox))
    return landmarks


def similarity_transform(input_points, output_points):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    input_points = np.copy(input_points).tolist()
    output_points = np.copy(output_points).tolist()

    # The third point is calculated so that the three points make an equilateral triangle
    x_input = c60 * (input_points[0][0] - input_points[1][0]) - s60 * (input_points[0][1] - input_points[1][1]) \
              + input_points[1][0]
    y_input = s60 * (input_points[0][0] - input_points[1][0]) + c60 * (input_points[0][1] - input_points[1][1]) \
              + input_points[1][1]

    input_points.append([np.int(x_input), np.int(y_input)])

    x_output = c60 * (output_points[0][0] - output_points[1][0]) - s60 * (output_points[0][1] - output_points[1][1]) \
               + output_points[1][0]
    y_output = s60 * (output_points[0][0] - output_points[1][0]) + c60 * (output_points[0][1] - output_points[1][1]) \
               + output_points[1][1]

    output_points.append([np.int(x_output), np.int(y_output)])

    transform = cv2.estimateAffinePartial2D(np.array([input_points]), np.array([output_points]))
    return transform[0]


def normalize_images_and_landmarks(output_size, input_image, input_points):
    h, w = output_size
    input_eyecorner = [input_points[36], input_points[45]]
    output_eyecorner = [(np.int(0.3 * w), np.int(h / 3)),
                        (np.int(0.7 * w), np.int(h / 3))]
    transform = similarity_transform(input_eyecorner, output_eyecorner)
    output_image = cv2.warpAffine(input_image, transform, (w, h))
    resized_input_points = np.reshape(input_points,
                                      (input_points.shape[0], 1, input_points.shape[1]))

    out_points = cv2.transform(resized_input_points, transform)

    resized_output_points = np.reshape(out_points,
                                       (input_points.shape[0], input_points.shape[1]))

    return output_image, resized_output_points


def get_eight_boundary_points(h, w):
    boundary_points = []
    boundary_points.append((0, 0))
    boundary_points.append((w / 2, 0))
    boundary_points.append((w - 1, 0))
    boundary_points.append((w - 1, h / 2))
    boundary_points.append((w - 1, h - 1))
    boundary_points.append((w / 2, h - 1))
    boundary_points.append((0, h - 1))
    boundary_points.append((0, h / 2))
    return np.array(boundary_points, dtype=np.float)


def landmarks_to_x_y(landmarks):
    points = []
    for i in range(0, 68):
        point = (landmarks.part(i).x, landmarks.part(i).y)
        points.append(point)
    return points


def resize_images(landmarks, presidents, h, w, boundary_points):
    numLandmarks = 68
    numImages = len(presidents)
    imagesNorm = []
    pointsNorm = []
    pointsAvg = np.zeros((numLandmarks, 2), dtype=np.float32)
    for i, image in enumerate(presidents):
        points = landmarks_to_x_y(landmarks[i])
        points = np.array(points)
        out_image, out_points = normalize_images_and_landmarks((h, w), image, points)
        pointsAvg = pointsAvg + (out_points / numImages)
        out_points = np.concatenate((out_points, boundary_points), axis=0)
        pointsNorm.append(out_points)
        imagesNorm.append(out_image)
    pointsAvg = np.concatenate((pointsAvg, boundary_points), axis=0)
    return pointsAvg, imagesNorm, pointsNorm


def check_points_in_rect(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def find_index(points, point):
    difference = np.array(points) - np.array(point)
    difference_norm = np.linalg.norm(difference, 2, 1)
    return np.argmin(difference_norm)


def write_delaunay(rect, points):
    dt = []
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((p[0], p[1]))
    triangle_list = subdiv.getTriangleList()
    for triangle in triangle_list:
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])
        if check_points_in_rect(rect, pt1) and check_points_in_rect(rect, pt2) and check_points_in_rect(rect, pt3):
            landmark1 = find_index(points, pt1)
            landmark2 = find_index(points, pt2)
            landmark3 = find_index(points, pt3)
            dt.append((landmark1, landmark2, landmark3))
    return dt


def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p


def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


def warpTriangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    t1Rect = []
    t2Rect = []
    t2RectInt = []
    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


def warp_image(imagesNorm, pointsNorm, ave_points, delaunay_triangles):
    h, w, ch = imagesNorm.shape
    imOut = np.zeros(imagesNorm.shape, dtype=imagesNorm.dtype)
    for j in range(0, len(delaunay_triangles)):
        tin = []
        tout = []
        for k in range(0, 3):
            pIn = pointsNorm[delaunay_triangles[j][k]]
            pIn = constrainPoint(pIn, w, h)

            pOut = ave_points[delaunay_triangles[j][k]]
            pOut = constrainPoint(pOut, w, h)

            tin.append(pIn)
            tout.append(pOut)

        warpTriangle(imagesNorm, imOut, tin, tout)
    return imOut


if __name__ == "__main__":
    # Read all images
    cwd = os.getcwd()
    presidents_path = cwd + '/images/presidents/my-im'
    presidents = [cv2.imread(file) for file in glob.glob(presidents_path + '/*.jpg')]

    # perform landmark detection using dlib
    landmarks = landmarks_detection(presidents)

    # Returns 8 points on the boundary of a rectangle
    h = 600
    w = 600
    boundary_points = get_eight_boundary_points(h, w)
    ave_points, imagesNorm, pointsNorm = resize_images(landmarks, presidents, h, w, boundary_points)

    # delaunay
    rect = (0, 0, w, h)
    delaunay_triangles = write_delaunay(rect, ave_points)

    # Output image
    output = np.zeros((h, w, 3), dtype=np.float)
    for i in range(0, len(presidents)):
        image_warp = warp_image(imagesNorm[i], pointsNorm[i], ave_points.tolist(), delaunay_triangles)
        output = output + image_warp

    output = output / (1.0 * len(presidents))

    # Display result
    plt.imshow(output[:, :, ::-1] / 255)
    plt.show()
