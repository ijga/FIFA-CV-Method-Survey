from ultralytics import YOLO
import numpy as np
import cv2
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

model = YOLO('best.pt')

# results = model(source="gameplays.mp4", show=True, conf=0.4, save=False, stream=True)
results = model(source="gameplays.mp4", stream=True, show=True)

Hs = np.array([[ 8.61212467e-01,  1.32034131e+00,  8.66279002e+01],
     [-1.85248628e-03,  3.25171615e+00, -5.49114561e+02],
     [-3.41393703e-06,  2.06444132e-03,  1.00000000e+00]])

field = imread('field/r_endfield.png')

for idx, result in enumerate(results):
    # print("NEW: ", idx)

    boxes = result.boxes.xywh.numpy()[:, :2] 
    boxes[:, 1] += (result.boxes.xywh.numpy()[:, 3] * 0.5)
    boxes = boxes.reshape(1, -1, 2)

    classes = result.boxes.cls.numpy()
    confidence = result.boxes.conf.numpy()

    points = cv2.perspectiveTransform(boxes, Hs)

    img = np.zeros((720, 1280, 3), dtype=np.uint8)  # Assuming 1280x720 resolution
    for type, point in zip(classes, points[0, :, :]):
        if type == 7:
            cv2.circle(img, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)  # Draw a red circle at each point
        elif type == 9:
            cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)  # Draw a red circle at each point
        else:
            cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Draw a red circle at each point

    cv2.imshow('game items', img)
