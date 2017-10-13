import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils import *

n = 100
features = 3
img_dims = [256.0,256.0]
X_test = np.random.rand(n,features)-.5
Y_test = np.random.rand(n,features)-.5
Y_pred = Y_test + 0.02*(np.random.rand(n,features)-.5)

#utils.plot_prediction(X_test, Y_test, Y_pred,img_dims)

filename = 'Test/steelpan_49990.txt'

#arrs = parse_txt_file(filename)

#print("arrs = ",arrs)



X, Y, img_dims, img_file_list = build_dataset(load_frac=1)
