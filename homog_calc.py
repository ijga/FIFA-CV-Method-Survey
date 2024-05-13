import numpy as np
import matplotlib.pyplot as plt

from skimage import transform
from skimage.io import imread, imshow
import cv2

left_end = imread('images/1080_default.png')
imshow(left_end) 
# plt.show()

# 1080_default
src = np.array([797, 221, # tl
                1126, 221, # tr
                534, 1077, # bl
                1381, 1077, # br
]).reshape((4, 2))

# Plot the points on the image with red color
plt.scatter(src[:, 0], src[:, 1], color='red', marker='o')

# 1080_default
dst = np.array([870, 150, # tl
                1030, 150, # tr
                870, 930, # bl
                1030, 930, # br
]).reshape((4, 2))

# # right_end
# src = np.array([807, 410, # tl
#                 1290, 383, # tr
#                 1171, 795, # bl
#                 1853, 741, # br
# ]).reshape((4, 2))



# # right_end
# dst = np.array([1500, 230, # tl
#                 1750, 230, # tr
#                 1500, 850, # bl
#                 1750, 850, # br
# ]).reshape((4, 2))

# # left_end
# src = np.array([617, 389, # tl
#                 1100, 416, # tr
#                 46, 748, # bl
#                 733, 805, # br
# ]).reshape((4, 2))

# # Plot the points on the image with red color
# plt.scatter(src[:, 0], src[:, 1], color='red', marker='o')

# # left_end
# dst = np.array([170, 230, # tl
#                 420, 230, # tr
#                 170, 850, # bl
#                 420, 850, # br
# ]).reshape((4, 2))

tform = transform.estimate_transform('projective', src, dst)

# Hr = np.array(
#     [[ 9.73749489e-01,  2.61432135e+00,  1.10399484e+03], # position 0, 2 is important here, this will align endlines when added to Hc
#      [ 2.12154164e-01,  4.35152471e+00, -1.40248615e+03], # position 1, 0 is important here, this will align sidelines when added to Hc
#      [ 1.96328735e-05,  2.33807617e-03,  1.00000000e+00]])

# Hr_correct = np.array(
#     [[ 9.73749489e-01,  2.61432135e+00,  1.10399484e+03],
#      [ 2.50914243e-01,  4.94555617e+00, -1.77604118e+03],
#      [ 1.96328735e-05,  2.33807617e-03,  1.00000000e+00]])

# Hc = np.array([[ 8.33519727e-01,  1.93684006e+00,  1.61288507e+02],
#      [ 1.43620104e-05,  4.30087726e+00, -1.22802498e+03 + 300],
#      [ 2.09970635e-08,  2.01857912e-03,  1.00000000e+00]])

# zero_two = 1.10399484e+03 - 1.61288507e+02
# # zero_two = 1.10399484e+03
# one_zero = 2.50914243e-01 - 1.43620104e-05
# y_offset = 200

# Hl_guess = np.array(
#     [[ 8.33519727e-01,  1.93684006e+00,  1.61288507e+02 - zero_two],
#      [ 1.43620104e-05 - one_zero,  4.30087726e+00, -1.22802498e+03 + y_offset],
#      [ 2.09970635e-08,  2.01857912e-03,  1.00000000e+00]]
# )

#     Hl = np.array([[ 9.60447704e-01,  1.95174080e+00, -1.01855334e+03],
#                    [-2.55089434e-01,  5.10054596e+00, -1.37582741e+03],
#                    [-8.15372620e-06,  2.48185855e-03,  1.00000000e+00]])
# print(Hl_guess)

# Hci = np.linalg.inv(Hc)

print(tform)
print(tform.inverse)

tf_img = transform.warp(left_end, tform.inverse) # in practice, use tform H matrix
fig, ax = plt.subplots()
# ax.scatter(dst[:, 0], dst[:, 1], color='red', marker='o')
ax.imshow(tf_img)
_ = ax.set_title('projective transformation')

plt.imshow(tf_img)
# # Plot the points on the image with red color
plt.scatter(dst[:, 0], dst[:, 1], color='red', marker='o')
plt.show()


