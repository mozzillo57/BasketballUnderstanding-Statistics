# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import glob
from math import sqrt

#pre-defined colors
match_color = ((0,0,0),(255,255,255),(255,0,0))

# Color Distance_Input:
# c: type(Tuple) - tuple of color ex:(0,0,0)
# return pre-defined closest color to input
def color_distance(c):
    r, g, b = c[0], c[1], c[2]
    color_diffs = []
    for color in match_color:
        cr, cg, cb = color[0], color[1], color[2] 
        color_diff = sqrt(abs(r - cr)**2 + abs(g - cg)**2 + abs(b - cb)**2)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]

# Average Color_Input:
# img: type(numpy.ndarray) - input image
# return predefined color closest to input image dominant color  

def main_colors(img):
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    
    return color_distance(tuple(palette[np.argmax(counts)].astype(int)))

# scan image and provide rect
#scan_img Input:
# img: type(numpy.ndarray) - input image
# filename: type(string) - image path
# return image with predicted bounding boxes

def scan_img(image, filename):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # detect people in the image
    rects, weights = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    copy_image = image.copy()

    pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)
    for (xA, yA, xB, yB) in pick:
        #for rects correlated to team color
        color = main_colors(copy_image[yA:yB, xA:xB])
        cv2.rectangle(image, (int(xA), int(yA)), (int(xB), int(yB)), [int(x) for x in color], 4)
        
        # for classical green rect
        #cv2.rectangle(image, (int(xA), int(yA)), (int(xB), int(yB)), (0,255,0), 3)
    
    # show some information on the number of bounding boxes
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))
    name = filename.split('/')[-1]
    cv2.imwrite(f'./output/{name}', image)
    return image

path = "./input/*.jpg"

for filepath in glob.iglob(path):
    img = cv2.imread(filepath)
    img_with_box = scan_img(img, filepath)
