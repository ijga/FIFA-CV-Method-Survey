import numpy as np
from  cv2 import imread, filter2D
import cv2
import matplotlib.pyplot as plt

from skimage import color
from skimage import io

img = io.imread('images/hough_midfield.png')

# img = imread("images/1080_right_end.png")
# img = img[:,:,[2, 1, 0]]

# plt.imshow(img)
# plt.show()



horiz_edge_kernel = np.array([[0.5, 0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-0.5, -0.5, -0.5, -0.5, -0.5]])
Z = filter2D(img,-1,horiz_edge_kernel)
R = np.max(Z[:, :, 0])
print(Z)

cv2.imshow("edge", Z)
cv2.waitKey(0)
cv2.destroyAllWindows()

# lines = cv2.HoughLines(Z, 1, np.pi, 200)
# print(lines)

# cv2.imshow("edge detect", Z)
RImg = Z[:,:,0]
GImg = Z[:,:,1]
BImg = Z[:,:,2]

Sb = np.bitwise_and((GImg>=50).astype(np.uint8), np.bitwise_and((RImg>=255).astype(np.uint8), (BImg>=255).astype(np.uint8)))

cv2.imshow("Sb", Sb*255)
cv2.waitKey(0)
cv2.destroyAllWindows()
