
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import cv2
import PIL

in_filename = 'Val/06240907_proc_01060.png'

img = load_img(in_filename)                 # this is a PIL image
img = img_to_array(img)
height, width, channels = img.shape
print("width, height, channels = ",width, height, channels)

img = np.flipud(img)

out_filename = "flipped.png"
cv2.imwrite(out_filename,img)
