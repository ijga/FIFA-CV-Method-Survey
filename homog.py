import numpy as np
import matplotlib.pyplot as plt

from skimage import transform
from skimage.io import imread, imshow
import cv2

field = imread('field/midfield2.png')
imshow(field) 
# plt.show()

# midfield
src = np.array([638, 303,
                638, 444,
                402, 369,
                875, 369,
                404, 171,
                1093, 719,
                186, 719,
875, 171]).reshape((8, 2))

# r_endfield

# Plot the points on the image with red color
plt.scatter(src[:, 0], src[:, 1], color='red', marker='o')

# midfield
dst = np.array([638, 303,
                638, 444,
                575, 369,
                702, 369,
                469, 0,
                810, 719,
                469, 719,
810, 0]).reshape((8, 2))

# r_endfield

tform = transform.estimate_transform('projective', src, dst)

# print(tform)
# print(tform.inverse)

tf_img = transform.warp(field, tform.inverse) # in practice, use tform H matrix
fig, ax = plt.subplots()
ax.imshow(tf_img)
_ = ax.set_title('projective transformation')

plt.imshow(tf_img)
plt.show()


