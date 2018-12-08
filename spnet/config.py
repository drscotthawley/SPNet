import numpy as np

# define some global parameters
dtype = np.float32

meta_extension = ".csv"   # extension for metadata files

# Define some colors: openCV uses BGR instead of RGB
blue = (255,0,0)
red = (0,0,255)
green = (0,200,0)
white = (255, 255, 255)
black = (0, 0, 0)
grey = (128, 128, 128)

# define indices for different parts of the data stream
vars_per_pred = 8
ind_cx = 0
ind_cy = 1
ind_semi_a = 2
ind_semi_b = 3
ind_angle1 = 4          # cos(2*theta)
ind_angle2 = 5          # sin(2*theta)
ind_noobj = 6
ind_rings = 7

loss_type = 'hybrid' #  'same'=mse for all, 'hybrid'(or anything !'same')=CE for noobj & mse for all others
