import cv2
import numpy as np
from segment import segment_by_angle_kmeans
from intersection import intersection, segmented_intersections
from draw_poly import draw_poly_box


def get_court(image):
    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 15
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imwrite('output/original.jpg', img)
    cv2.imwrite('output/filteredimgs/lines.jpg', lines_edges)
    cv2.imwrite('output/filteredimgs/edges.jpg', edges)

    lower_threshold = 100
    upper_threshold = 150
    threshold = 400

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    num_lines = 3

    if len(lines) > num_lines:
        while(len(lines) > num_lines):
            lower_threshold = lower_threshold + 1
            upper_threshold = upper_threshold + 1
            threshold = threshold + 1
            edges = cv2.Canny(blur_gray, lower_threshold, upper_threshold)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    else:
        while(len(lines) < num_lines):
            threshold = threshold - 1
            edges = cv2.Canny(gray, lower_threshold, upper_threshold)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    # convert each line to coordinates back in the original image

    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 10000 * -b)
            y1 = int(y0 + 10000 * a)
            x2 = int(x0 - 10000 * -b)
            y2 = int(y0 - 10000 * a)

            # draw each line on the image
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imwrite('output/filteredimgs/houghlines.jpg', img)

    segmented = segment_by_angle_kmeans(lines)
    # get LEFT TOP and LEFT BOTTOM point
    intersections = segmented_intersections(segmented)

    p1 = img.shape[1] - intersections[0][0][0]
    p2 = img.shape[1] - intersections[1][0][1]

    point1 = [[p1, intersections[1][0][1]]]  # RIGHT TOP
    point2 = [[p2, intersections[0][0][1]]]  # RIGHT BOTTOM

    # Order the intersection
    intersections.insert(1, point2)
    intersections.insert(2, point1)

    # remove the third dimension of the intersections
    intersections = np.squeeze(intersections)

    # draw the points of the court on image
    for n in range(len(intersections)):
        image = cv2.circle(img, (intersections[n][0], intersections[n][1]), radius=0, color=(
            255, 0, 0), thickness=15)
    cv2.imwrite('output/dots.jpg', image)

    final = draw_poly_box(img, np.array(intersections))
    cv2.imwrite('output/court3D_detection.jpg', final)

    return intersections
