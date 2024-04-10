# Python program to illustrate HoughLine
# method for line detection
import cv2
import numpy as np

def print_info(r="_", theta="_", x0="_", y0="_", x_bottom="_", goal="_", x1="_", y1="_", x2="_", y2="_", img_width=0):
	print("	")
	print(f'r, theta: ({r},{theta})')
	print(f'x0, y0: ({x0}, {y0}')
	print(f'x_bottom: {x_bottom}, goal:{goal}')
	print(f'endpoints: ({x1}, {y1}), ({x2}, {y2})')
	print(f'distance from horiz edges: {abs(x1)}, {abs(x2-img_width)}')

img = cv2.imread('images/mid2.png')
img_height, img_width, color_dims = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert img to grayscale


gray_blur = cv2.GaussianBlur(gray, (35, 35), 0)
# cv2.imshow("gray", gray_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


horiz_edge_kernel = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, -1, -1, -1, -1]])
edges = cv2.filter2D(gray_blur,-1,horiz_edge_kernel)
# cv2.imshow("edges1", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
edges = (edges>=30).astype(np.uint8) * 255
# cv2.imshow("edges2", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
edges = cv2.filter2D(edges,-1,horiz_edge_kernel)
# cv2.imshow("edges3", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# This returns an array of r and theta values
lines = cv2.HoughLines(edges, 1, np.pi/180, 600, min_theta=1.48353, max_theta=1.65806)

print(len(lines))

# The below for loop runs till r and theta values
# are in the range of the 2d array
for r_theta in lines:
	arr = np.array(r_theta[0], dtype=np.float64)
	r, theta = arr
	# print(r, theta)

	x0 = np.cos(theta)*r
	y0 = np.sin(theta)*r	
	
	too_low = 0.45388889
	top_third = 0.35759259259

	max_expected_bounds = 30000 # 1000
	
	x_bottom = (y0 - (top_third) * img_height)/np.cos(theta) # where line intersects line (y=top_third)
	goal = (x0 - x_bottom) # where x should be set to have endpoint at x_bottom

	if theta < 1.5708: # right side
	
		x1 = int(x0 + goal * (-np.sin(theta))) 
		y1 = int(y0 + goal * (np.cos(theta)))

		x2 = int(x0 - 1000 * (-np.sin(theta))) 
		y2 = int(y0 - 1000 * ( np.cos(theta)))

		if abs(x1) > max_expected_bounds:
			print(x1, max_expected_bounds)
			continue
			x1 = int(x0 + 1000 * (-np.sin(theta))) 
			y1 = int(y0 + 1000 * (np.cos(theta)))
			print_info(r, theta, x0, y0, x_bottom, goal, x1, y1, x2, y2, img_width)
		else:
			print_info(r, theta, x0, y0, x_bottom, goal, x1, y1, x2, y2, img_width)
			pass

	elif theta > 1.5708: # left side

		x1 = int(x0 - 1000 * (-np.sin(theta))) 
		y1 = int(y0 - 1000 * (np.cos(theta)))

		x2 = int(x0 + goal * (-np.sin(theta))) 
		y2 = int(y0 + goal * ( np.cos(theta)))
		
		if abs(x2-img_width) > max_expected_bounds: # outside expected bounds
			x2 = int(x0 - 1000 * (-np.sin(theta))) 
			y2 = int(y0 - 1000 * (np.cos(theta)))
			print_info(r, theta, x0, y0, x_bottom, goal, x1, y1, x2, y2, img_width)			
		else:
			print_info(r, theta, x0, y0, x_bottom, goal, x1, y1, x2, y2, img_width)	
			pass

	if y1 > too_low * img_height or y2 > too_low * img_height:
		continue
	cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()