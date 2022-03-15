from cv2 import cv2
import numpy as np
import random

''' Delaunay Triangulation'''


def find_index(points, point):
    difference = np.array(points) - np.array(point)
    difference_norm = np.linalg.norm(difference, 2, 1)
    return np.argmin(difference_norm)


def write_delaunay(subdiv, points, outputFileName):
    triangle_list = subdiv.getTriangleList()
    filePointer = open(outputFileName, 'w')
    for triangle in triangle_list:
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])
        landmark1 = find_index(points, pt1)
        landmark2 = find_index(points, pt2)
        landmark3 = find_index(points, pt3)
        filePointer.write("{},{},{}\n".format(landmark1, landmark2, landmark3))
    filePointer.close()


def check_points_in_rect(rect, point):
    if point[0] < rec[0]:
        return False
    elif point[1] < rec[1]:
        return False
    elif point[0] > rec[2]:
        return False
    elif point[1] > rec[3]:
        return False
    return True


def draw_delaunay(img, subdiv, delaunay_color, rect):
    triangle_list = subdiv.getTriangleList()
    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if check_points_in_rect(rect, pt1) and check_points_in_rect(rect, pt2) and check_points_in_rect(rect, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


def drawPoint(image, point, color):
    cv2.circle(image, point, 2, color, -1, cv2.LINE_AA, 0)


def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacetArr = []
        for f in facets[i]:
            ifacetArr.append(f)
        ifacet = np.array(ifacetArr, np.int)

        color = (random.randint(0, 255),
                 random.randint(0, 255),
                 random.randint(0, 255))
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);

        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]),
                   3, (0, 0, 0), -1, cv2.LINE_AA, 0)



if __name__ == "__main__":

    image = cv2.imread("images/smiling-man.jpg")
    points = []
    plotPoints = []
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)
    img_voronoi = np.zeros(image.shape, dtype=image.dtype)
    video = cv2.VideoWriter("results/delaunay_voronoi.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2,
                            (2 * image.shape[0], image.shape[1]))
    outputFileName = "results/smiling-man-delaunay.tri"

    # Read in the points from a text file
    with open("images/smiling-man-delaunay.txt") as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    # Rectangle to be used with Subdiv2D
    rec = (0, 0, image.shape[1], image.shape[0])
    subdiv = cv2.Subdiv2D(rec)

    # Draw landmark points on the image
    for point in points:
        drawPoint(image, point, points_color)

    # Insert points into subdiv
    for point in points:
        subdiv.insert(point)
        plotPoints.append(point)
        img_delaunay = image.copy()
        draw_delaunay(img_delaunay, subdiv, delaunay_color, rec)
        draw_voronoi(img_voronoi, subdiv)

        for pp in plotPoints:
            drawPoint(img_delaunay, pp, points_color)

        combined = cv2.hconcat([img_delaunay, img_voronoi])
        video.write(combined)
    video.release()

    write_delaunay(subdiv, points, outputFileName)
    print("Writing Delaunay triangles to {}".format(outputFileName))
