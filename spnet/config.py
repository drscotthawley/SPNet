import numpy as np

# define some global parameters
dtype = np.float32

meta_extension = ".csv"   # extension for metadata files

# Define some colors:
# Note! These are all "backward" because for some reason openCV uses BGR instead of RGB
#  Can use "[::-1]" slicing below in order to read it RGB-style
blue = (255,0,0)
red = (0,0,255)
green = (0,200,0)
white = (255, 255, 255)
black = (0, 0, 0)
grey = (128,128,128)
lightgrey = (210, 210, 210)
yellow = (255,255,0)[::-1]
cyan = (0,220,220)[::-1]
mpl_blue = (31, 140, 200)[::-1]  #matplotlib blue...only brigter
mpl_orange = (255, 127, 14)[::-1] # matplotlib orange..only brighter
veridis_purple = (72, 18, 84)[::-1] # dark purple from veridis color map
veridis_lightgreen = (97, 207, 99)[::-1] # light green I picked out
veridis_yellow = (254, 228, 76)[::-1]
magma_light = (253, 252, 197)[::-1] # light end of magma colormap
truecolor = yellow
predcolor = veridis_purple

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

loss_type = 'same' #  'same'=mse for all, 'hybrid'(or anything !'same')=CE for noobj & mse for all others

# model_type:   mostly defines what NOT to do
#              'monolithic' or '<any string not on this list>': resize images to 331x331
#              'simple': don't use functional API.  Not recommended!
#              'compound': old interface where we reorder outputs  Not recommended!
#               'ss': include a SelectiveSigmoid layer
#              'big':  do not resize input images at all.
model_type = 'monolithic'

# name of a predefined Keras model
#   'Xception' works well, 'InceptionResNetV2' seems too big & slow, 'NASNetLarge' won't fit
basemodel = 'Xception'
