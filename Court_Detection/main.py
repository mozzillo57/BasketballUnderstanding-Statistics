import torch
import cv2
import numpy as np
from court import get_court

model = torch.hub.load('./yolov5', 'custom',
                       path='./yolov5/runs/train/exp2/weights/best.pt', source='local')

# in alternative we can load the model online and pass only the weights
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

model.conf = 0.45  # confidence threshold (0-1)
# Image
img_path = 'input/bb.png'
court_path = 'input/court.jpg'
# Inference
img_clean = cv2.imread(img_path)
img_dst = cv2.imread(court_path)
results = model(img_clean)

# Manually insert four corners of the court
dst_pts = np.array([
    [10,  350],  # LEFT BOTTOM
    [315, 350],  # RIGHT BOTTOM
    [315,  10],   # TOP RIGHT
    [10,    10],   # TOP LEFT
])

# save the rectangle of the court inserted manually
cv2.imwrite('output/court_manual.jpg', cv2.polylines(img_dst.copy(),
            [dst_pts], isClosed=True, color=[255, 0, 0], thickness=2))

# get the coordinates of predicted bounding boxes
pred_boxes = results.xyxy[0]
pred_boxes = pred_boxes.detach().cpu().numpy()
pred_boxes = np.delete(pred_boxes, 4, 1)

print(pred_boxes)

# tke only value of the bounding boxes
pred_boxes = pred_boxes.astype(int)

# get src_points
intersections = get_court(img_clean.copy())
src_pts = np.array(intersections, np.float32)

im = img_clean.copy()
color = [255, 0, 0]
thickness = 10
radius = 1

for box in pred_boxes:
    # Include only class player1 and player2
    cls = box[4]
    if cls <= 1:
        x1, y1, x2, y2 = box[:4]
        # calculate the x of center
        xc = x1 + int((x2 - x1)/2)
        # calculate position of the players based on xc
        pos1 = (xc - 1, y2)
        pos2 = (xc + 1, y2 + 1)
        cv2.rectangle(im, pos1, pos2, color, thickness)

cv2.imwrite('output/player_detected.jpg', im)

# Finds the perspective transformation H between the 3d court and the 2D court
h, _ = cv2.findHomography(src_pts, dst_pts)
img_out = cv2.warpPerspective(im, h, (img_dst.shape[1], img_dst.shape[0]))

cv2.imwrite('output/wraped_img.jpg', img_out)

# isolate the blue point for get the coordinates
# Set the Lower range value of blue in BGR
lower_range = np.array([255, 0, 0])
# Set the Upper range value of blue in BGR
upper_range = np.array([255, 155, 155])
# Create a mask with range
mask = cv2.inRange(img_out.copy(), lower_range, upper_range)
result = cv2.bitwise_and(img_out.copy(), img_out.copy(), mask=mask)
cv2.imwrite('output/filteredimgs/filtered_points.jpg', result)

cnts, _ = cv2.findContours(
    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
court = img_dst.copy()
for cnt in cnts:
    for c in cnt:
        # print(c[0])
        for pos in c:
            #print(pos[0], pos[1])
            center_coordinates = (pos[0], pos[1])
            cv2.circle(court, center_coordinates, radius=0,
                       color=(0, 0, 0), thickness=15)
cv2.imwrite('output/court_2D.jpg', court)

print('Completed.')
