from ultralytics import YOLO
import numpy as np
import cv2
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

model = YOLO('models/best.pt')
np.set_printoptions(threshold = np.inf)

def print_info(r="_", theta="_", x0="_", y0="_", x_bottom="_", goal="_", x1="_", y1="_", x2="_", y2="_", img_width=0):
	print("	")
	print(f'r, theta: ({r},{theta})')
	print(f'x0, y0: ({x0}, {y0}')
	print(f'x_bottom: {x_bottom}, goal:{goal}')
	print(f'endpoints: ({x1}, {y1}), ({x2}, {y2})')
	print(f'distance from horiz edges: {abs(x1)}, {abs(x2-img_width)}')

# results = model(source="gameplays.mp4", show=True, conf=0.4, save=False, stream=True)
results = model(source="videos/g1_5-5_2024-04-11.mp4", stream=True, show=True)

# 1080
Hs = np.array([[ 8.19403396e-01,  2.94796166e+00,  1.61297429e+02],
     [-2.16531130e-16,  4.41745540e+00, -7.23522908e+02],
     [ 5.23343066e-19,  3.09908703e-03,  1.00000000e+00]])

last_theta = 0

for idx, result in enumerate(results):
    boxes = result.boxes.xywh.numpy()[:, :2] 
    boxes[:, 1] += (result.boxes.xywh.numpy()[:, 3] * 0.5)
    boxes = boxes.reshape(1, -1, 2)

    classes = result.boxes.cls.numpy()
    confidence = result.boxes.conf.numpy()
    
    points = np.array([])
    try:
        points = cv2.perspectiveTransform(boxes, Hs)
        if len(points) > 0: # error thrown if null
            pass
    except:
        points = np.array([])

    img = np.ones((1080, 1920, 3), dtype=np.uint8)  # 1080p
    img = 200 * img
    if len(points) > 0:
        for type, point in zip(classes, points[0, :, :]):
            if type == 7:
                cv2.circle(img, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)  # Draw a red circle at each point
            elif type == 9:
                cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)  # Draw a red circle at each point
            else:
                cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Draw a red circle at each point

    cv2.imshow('game items', img)
