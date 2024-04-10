import numpy as np
import matplotlib.pyplot as plt

from skimage import transform
from skimage.io import imread, imshow
import cv2

left_end = imread('images/1080_left_end.png')
imshow(left_end) 
# plt.show()

# left_end
src = np.array([807, 410, # tl
                1290, 383, # tr
                1171, 795, # bl
                1853, 741, # br
]).reshape((4, 2))

# Plot the points on the image with red color
plt.scatter(src[:, 0], src[:, 1], color='red', marker='o')

# left_end
dst = np.array([1500, 280, # tl
                1750, 280, # tr
                1500, 800, # bl
                1750, 800, # br
]).reshape((4, 2))

tform = transform.estimate_transform('projective', src, dst)

Hc = np.array([[ 8.33519727e-01,  1.93684006e+00,  1.10399484e+03], # position 0, 2 is important here, this will align endlines
     [ 2.12154164e-01,  4.30087726e+00, -1.22802498e+03], # position 1, 0 is important here, this will align sidelines
     [ 2.09970635e-08,  2.01857912e-03,  1.00000000e+00]])


zero_two = 1.10399484e+03 - 1.61288507e+02
one_zero = 2.12154164e-01 - 1.43620104e-05
Hl_guess = np.array(
    [[ 8.33519727e-01,  1.93684006e+00,  1.61288507e+02 - zero_two],
     [ 1.43620104e-05 - one_zero,  4.30087726e+00, -1.22802498e+03 + 250],
     [ 2.09970635e-08,  2.01857912e-03,  1.00000000e+00]]
)

Hci = np.linalg.inv(Hl_guess)

print(tform)
print(tform.inverse)

tf_img = transform.warp(left_end, Hci) # in practice, use tform H matrix
fig, ax = plt.subplots()
ax.scatter(dst[:, 0], dst[:, 1], color='red', marker='o')
ax.imshow(tf_img)
_ = ax.set_title('projective transformation')

plt.imshow(tf_img)
plt.show()


