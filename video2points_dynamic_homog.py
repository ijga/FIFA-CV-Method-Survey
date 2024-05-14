from ultralytics import YOLO
import numpy as np
import cv2
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import time

model = YOLO('models/best.pt')
np.set_printoptions(threshold = np.inf)

file1 = open("points/points.txt", "a")  # append mode

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

# 720p
Hs = np.array([[ 8.61212467e-01,  1.32034131e+00,  8.66279002e+01],
     [-1.85248628e-03,  3.25171615e+00, -5.49114561e+02],
     [-3.41393703e-06,  2.06444132e-03,  1.00000000e+00]])

def calcHomog(theta):

    even = 1.5707964897155762
    goalkick_offset = 0.0436332
    value = theta - even
    print(value)

    x1 = 73.655650176915*(value)**2 + 0.833519727
    x2 = 355.84689145836*(value)**2 + 1.93684006
    # tx = -24322.6*(value) + 82.2433
    tx = 1.61288507e+02
    # y1 = -5.79838*(value) + -0.00138694
    y1 =  1.43620104e-05
    y2 = 338.61744892211*(value)**2 + 4.30087726
    ty = -287845.3827689*(value)**2 - 1228.02498
    # p1 = -0.000318411*(value) + 0.00000383338
    p1 = 2.09970635e-08
    p2 = 0.16781575188979*(value)**2 + 0.00201857912

    Hr = np.array([[ 9.73749489e-01,  2.61432135e+00,  1.10399484e+03],
                   [ 2.50914243e-01,  4.94555617e+00, -1.77604118e+03],
                   [ 1.96328735e-05,  2.33807617e-03,  1.00000000e+00]]) # -2.5 degrees
    
    Hc = np.array([[ 8.33519727e-01,  1.93684006e+00,  1.61288507e+02],
                   [ 1.43620104e-05,  4.30087726e+00, -1.22802498e+03],
                   [ 2.09970635e-08,  2.01857912e-03,  1.00000000e+00]])

    Hl = np.array([[ 9.60447704e-01,  1.95174080e+00, -1.01855334e+03],
                   [-2.55089434e-01,  5.10054596e+00, -1.37582741e+03],
                   [-8.15372620e-06,  2.48185855e-03,  1.00000000e+00]]) # +2.5 degrees

    # [quadratic, quadratic, linear
    # linear, quadratic, quadratic
    # linear, quadratic, constant]

    homog = np.array([[ x1, x2, tx],
                        [y1, y2, ty],
                        [p1,  p2,  1.00000000e+00]])

    return homog

last_theta = 0

endem = time.perf_counter()

for idx, result in enumerate(results):

    start = time.perf_counter()
    print(f'across: {(start - endem) * 1000}')
    

    img = result.orig_img
    img_height, img_width = result.orig_shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    blue_blur = cv2.GaussianBlur(blue, (35, 35), 0) # blur
    green_blur = cv2.GaussianBlur(green, (35, 35), 0) # blur
    red_blur = cv2.GaussianBlur(red, (35, 35), 0) # blur

    # bottom_edge_kernel = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    #                                   [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
    #                                   [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    #                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                               [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], 
    #                               [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], 
    #                               [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]])
    bottom_edge_kernel = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1],
                                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1], 
                                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1],
                                # [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [-0.1, -0.2, -0.3, -0.4, -0.5, -0.4, -0.3, -0.2, -0.1], 
                                [-0.1, -0.2, -0.3, -0.4, -0.5, -0.4, -0.3, -0.2, -0.1], 
                                [-0.1, -0.2, -0.3, -0.4, -0.5, -0.4, -0.3, -0.2, -0.1]])
    bl_edges = cv2.filter2D(blue_blur,-1,bottom_edge_kernel)
    gn_edges = cv2.filter2D(green_blur,-1,bottom_edge_kernel)
    rd_edges = cv2.filter2D(red_blur,-1,bottom_edge_kernel) # first set of edge detection

    # cv2.imshow("blue edges", bl_edges)
    # cv2.imshow("green edges", gn_edges)
    # cv2.imshow("red edges", rd_edges)
    max_edges = np.maximum.reduce([bl_edges, gn_edges, rd_edges])
    # cv2.imshow("max edges", max_edges)

    mean = np.mean(max_edges[max_edges > 80]) # mean is around 171 to start
    edges = (max_edges>=mean).astype(np.uint8) * 255 # thresholding/segmenting
    # cv2.imshow("edges", edges)

    thinning_edge_kernel = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                                    # [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], 
                                    [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]])
    edges = cv2.filter2D(edges,-1, thinning_edge_kernel) # second set of edge detection to narrow lines
    # cv2.imshow("edges3", edges)

    lines = np.array([])
    linesDetected = False

    thresholds_to_try = [1200, 1150, 1100, 1050, 1000, 950, 900, 850, 800, 750, 700, 600, 500, 400, 300, 200, 100]
    thresh_idx = 0

    while thresh_idx < 17:
        try:
            lines = cv2.HoughLines(edges, 1, np.pi/180, thresholds_to_try[thresh_idx], min_theta=1.48353, max_theta=1.65806) # can't be more than 5 degrees of horizontal
            if len(lines) > 0: # lines have been found
                break
        except:
            lines = np.array([])
        thresh_idx += 1

    # print(thresholds_to_try[thresh_idx])
    theta_sum = 0
    r_sum = 0
    theta_count = 0

    if lines.any():
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            theta_sum += theta
            r_sum += r
            theta_count += 1
    else:
        lines = []
    
    theta = 0
    if len(lines) > 0:
        theta = theta_sum/theta_count
        r = r_sum/theta_count
        x0 = np.cos(theta) * r
        y0 = np.sin(theta) * r

        x1 = int(x0 + 1000 * (-np.sin(theta))) 
        y1 = int(y0 + 1000 * (np.cos(theta)))
        x2 = int(x0 - 1000 * (-np.sin(theta))) 
        y2 = int(y0 - 1000 * ( np.cos(theta)))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("lines", img)  

    if theta == 0:
        theta = last_theta
    else:
        last_theta = theta

    boxes = result.boxes.xywh.numpy()[:, :2] 
    boxes[:, 1] += (result.boxes.xywh.numpy()[:, 3] * 0.5)
    boxes = boxes.reshape(1, -1, 2)

    classes = result.boxes.cls.numpy()
    confidence = result.boxes.conf.numpy()
    
    points = np.array([])
    theta = 1.5707964897155762
    try:
        homog = calcHomog(theta)
        points = cv2.perspectiveTransform(boxes, homog)
        if len(points) > 0: # error thrown if null
            pass
    except:
        points = np.array([])



    end = time.perf_counter()
    print(f'within: {(end-start)*1000}')
    endem = end



    # img = np.ones((720, 1280, 3), dtype=np.uint8)  # Assuming 1280x720 resolution
    # img = np.ones((1080, 1920, 3), dtype=np.uint8)  # 1080p
    # img = 200 * img
    # if len(points) > 0:
    #     for type, point in zip(classes, points[0, :, :]):
    #         if type == 7 or type == 9:
    #              file1.write(f'{type} + ({point[0]}, {point[1]})\n')

    #         if type == 7:
    #             cv2.circle(img, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)  # Draw a red circle at each point
    #         elif type == 9:
    #             cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)  # Draw a red circle at each point
    #         else:
    #             cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Draw a red circle at each point

    # cv2.imshow('game items', img)
    # cv2.waitKey(0)

    # file1.write(f'\n')

file1.close()
