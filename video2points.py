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

def plot_histogram(image, title, mask=None):
	# split the image into its respective channels, then initialize
	# the tuple of channel names along with our figure for plotting
	chans = cv2.split(image)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title(title)
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and plot it
		hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
		plt.plot(hist, color=color)
		plt.xlim([0, 256])

# results = model(source="gameplays.mp4", show=True, conf=0.4, save=False, stream=True)
results = model(source="videos/gameplays.mp4", stream=True, show=True)

Hs = np.array([[ 8.61212467e-01,  1.32034131e+00,  8.66279002e+01],
     [-1.85248628e-03,  3.25171615e+00, -5.49114561e+02],
     [-3.41393703e-06,  2.06444132e-03,  1.00000000e+00]])

for idx, result in enumerate(results):
    img = result.orig_img
    img_height, img_width = result.orig_shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    gray_blur = cv2.GaussianBlur(gray, (51, 51), 0) # blur
    print(gray_blur.max(), gray_blur.min())
    gray_blur = (255 * ((gray_blur - gray_blur.min()) / (gray_blur.max() - gray_blur.min()))).astype(np.uint8) # span histogram
    # print(gray_blur)
    cv2.imshow("gray", gray_blur)

    blue_blur = cv2.GaussianBlur(blue, (35, 35), 0) # blur
    green_blur = cv2.GaussianBlur(green, (35, 35), 0) # blur
    red_blur = cv2.GaussianBlur(red, (35, 35), 0) # blur

    # cv2.imshow("gray", gray_blur)
    gray_bottom_edge_kernel = np.array([
                                  [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], 
                                  [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], 
                                  [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                      [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                                      [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]])

    bottom_edge_kernel = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                      [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                                      [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], 
                                  [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], 
                                  [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]])
    gy_edges = cv2.filter2D(gray_blur,-1,gray_bottom_edge_kernel)
    bl_edges = cv2.filter2D(blue_blur,-1,bottom_edge_kernel)
    gn_edges = cv2.filter2D(green_blur,-1,bottom_edge_kernel)
    rd_edges = cv2.filter2D(red_blur,-1,bottom_edge_kernel) # first set of edge detection

    cv2.imshow("gray edges", gy_edges)
    # cv2.imshow("blue edges", bl_edges)
    # cv2.imshow("green edges", gn_edges)
    # cv2.imshow("red edges", rd_edges)
    max_edges = np.maximum.reduce([gy_edges, bl_edges, gn_edges, rd_edges])
    # cv2.imshow("max edges", max_edges)

    mean = np.mean(max_edges[max_edges > 80]) # mean is around 171 to start
    edges = (max_edges>=mean).astype(np.uint8) * 255 # thresholding/segmenting
    # cv2.imshow("edges", edges)

    edges = cv2.filter2D(edges,-1,bottom_edge_kernel) # second set of edge detection
    cv2.imshow("edges3", edges)


    lines = np.array([])
    linesDetected = False

    thresholds_to_try = [1000, 950, 900, 850, 800, 750, 700, 600, 500, 400, 300, 200, 100]
    thresh_idx = 0

    while thresh_idx < 10:
        try:
            lines = cv2.HoughLines(edges, 1, np.pi/180, thresholds_to_try[thresh_idx], min_theta=1.48353, max_theta=1.65806) # can't be more than 5 degrees of horizontal
            if len(lines) > 0: # lines have been found
                break
        except:
            lines = np.array([])
        thresh_idx += 1

    print(thresholds_to_try[thresh_idx])

    good_lines = []
    if lines.any():
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            x0 = np.cos(theta)*r
            y0 = np.sin(theta)*r	

            x1 = int(x0 + 1000 * (-np.sin(theta))) 
            y1 = int(y0 + 1000 * (np.cos(theta)))

            x2 = int(x0 - 1000 * (-np.sin(theta))) 
            y2 = int(y0 - 1000 * ( np.cos(theta)))

            good_lines.append((x1, y1, x2, y2))
    else:
        lines = []
    
    for good_line in good_lines:
        x1, y1, x2, y2 = good_line
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("lines", img)

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

    img = np.ones((720, 1280, 3), dtype=np.uint8)  # Assuming 1280x720 resolution
    img = 200 * img
    if len(points) > 0:
        for type, point in zip(classes, points[0, :, :]):
            if type == 7:
                cv2.circle(img, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)  # Draw a red circle at each point
            elif type == 9:
                cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)  # Draw a red circle at each point
            else:
                cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Draw a red circle at each point

    # cv2.imshow('game items', img)
