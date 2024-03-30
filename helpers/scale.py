# Original image dimensions
original_width = 2528
original_height = 1276

# Target image dimensions
target_width = 1280
target_height = 720

# Calculate scaling factors
width_scale = target_width / original_width
height_scale = target_height / original_height

# Original coordinates
coordinates = [
    (1262, 537), # midcircle
                (1262, 787), 
                (795, 654), 
                (1729, 654),
                (798, 304), # lines
                (2160, 1275),
                (368, 1275), 
                (1729, 304), 
 ]


# Scale coordinates
scaled_coordinates = [(int(x * width_scale), int(y * height_scale)) for x, y in coordinates]

# Print scaled coordinates
for coord in scaled_coordinates:
    print(coord)
