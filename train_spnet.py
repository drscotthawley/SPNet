#! /usr/bin/env python3
from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard
import time
from distutils.version import LooseVersion
import PIL
import io
from models import *
from utils import *
from multi_gpu import *

from scipy.optimize import curve_fit

def calc_errors(Yp, Yv):
    """
    Calc errors in ring counts
      Yp = model prediction
      Yv = true value for validation set
      Yp & Yv have already been 'denormalized' at this point

      Not: index = 2 is where the ring count is stored.
    """
    max_pred_antinodes = int(Yv.shape[1]/vars_per_pred)
    diff = Yp - Yv
    # pixel error in antinode centers
    pix_err = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
    ipem = np.argmax(pix_err)               # index of pixel error maximum

    # number of ring miscounts
    miscounts = 0
    total_obj = 0                           # total is the number of true objects
    for j in range(Yv.shape[0]):
        for an in range(max_pred_antinodes):
            ind = ind_rings + an * vars_per_pred
            rings_t = int(round(Yv[j,ind]))
            i_noobj = ind_noobj + an * vars_per_pred
            if (0 == int(round(Yv[j,i_noobj])) ):   # Is there supposed to be an object there? If so, count the rings
                total_obj += 1
                if (int(round(Yp[j,ind])) != rings_t): # compare integer ring counts
                    miscounts += 1
            elif (int(round(Yv[j,i_noobj])) != int(round(Yp[j,i_noobj]))):  # consider false background as a mistake
                miscounts += 1


    return miscounts, total_obj, pix_err, ipem


def acc_extrap_func(x, a, b, c):  # extrapolation function for accuracy
    return a * (1.0 - np.exp(-b * (x-c)))


# Custom callbacks

hist = []
train_loss_hist = []
val_loss_hist = []
my_val_loss_hist = []
acc_hist = []
center_loss_hist = []
size_loss_hist = []
angle_loss_hist = []
noobj_loss_hist = []
class_loss_hist = []

global_count = 0                	       # global_count is handy when nb_epoch = 1
n_epochs_per_plot = 1
class MyProgressCallback(Callback):      # Callbacks essentially get inserted into code at model.fit()
    """
    This is a callback routine for visualizing progress of training
    """
    def __init__(self, X_val=None, Y_val=None, val_file_list=None,
        #X_test=None, Y_test=None, test_file_list=None,
        log_dir="./logs", use_tb=False, pred_shape=[3,3,4,6]):
        self.X_val = X_val
        self.Y_val = Y_val
        self.val_file_list = val_file_list
        self.buf = io.BytesIO()
        self.log_dir = log_dir
        self.use_tb = use_tb              # use tensorboard
        #self.max_pred_antinodes=max_pred_antinodes
        self.pred_shape = pred_shape

    def on_train_begin(self, logs={}):
        hist = []
        train_loss_hist = []
        val_loss_hist = []
        my_val_loss_hist = []
        acc_hist = []

        if (self.use_tb):
            self.sess = K.get_session()
            self.writer = tf.summary.FileWriter(self.log_dir)

    def make_obj_centroid_list(self, Y, num_centroids, vars_per_pred):
        """
        This constructs a list of centroids of actual objects (noobj==0), up to a certain limit (num_centroids)
        Y: a grid of predictors, as in either Yv or Yp.  see on_epoch_end(), below
        num_centroids: The number of centroids to try to grab for plotting below
        """
        print('    Making list of real centroids...')
        xlist, ylist = [], []
        row, maxrow = 0, 1000
        while (len(xlist) < num_centroids) and (row < maxrow):
            # our predictors are arranged as multiple groups of columns (vars_per_pred in each group)
            for an in range( int(Y.shape[1]/vars_per_pred)):   # loop over all possible antinodes in array of receptors
                ind = ind_cx + an * vars_per_pred   # index of x position of this antinode
                ind_noobj = ind + 6
                if (Y[row,ind_noobj] < 0.4):   # object (probably) exists
                    xlist.append(Y[row,ind])
                    ylist.append(Y[row,ind+1])
            row += 1
        return xlist, ylist


    def on_epoch_end(self, epoch, logs={}):
        global global_count                 # we use global_count instead of epoch # b/c of "Outer" loop
        global_count = global_count+1

        #self.count = self.count + 1
        if (0 == global_count % n_epochs_per_plot):  # only do this stuff every n_epochs_per_plot
            hist.append(global_count)
            train_loss_hist.append(logs.get('loss'))
            val_loss_hist.append(logs.get('val_loss'))

            X_val = self.X_val
            Y_val = self.Y_val
            val_file_list = self.val_file_list
            img_dims = X_val[0].shape

            print("\n MyProgressCallback: predicting, testing & saving plot")  # basically do nothing
            #predict_and_test(model, X_test, Y_test, img_dims, test_file_list, hist=hist, tl_hist=train_loss_hist, vl_hist=val_loss_hist,)
            m = Y_val.shape[0]

            print("    Predicting... (m = ",m," frames in val set)",sep="")
            start_time = time.time()
            Y_pred = self.model.predict(X_val)
            elapsed = time.time() - start_time
            print("    ...elapsed time to predict = ",elapsed,"s.   FPS = ",m*1.0/elapsed)

            # detailed loss analysis, component by component
            my_val_loss, loss_parts = my_loss(Y_val, Y_pred)
            [center_loss, size_loss, angle_loss, noobj_loss, class_loss] = loss_parts
            my_val_loss_hist.append(my_val_loss)
            center_loss_hist.append(center_loss)
            size_loss_hist.append(size_loss)
            angle_loss_hist.append(angle_loss)
            noobj_loss_hist.append(noobj_loss)
            class_loss_hist.append(class_loss)

            # calc some errors.  first transform back into regular 'world' values via de-normalization
            Yv = denorm_Y(Y_val)     # t is for true
            Yp = denorm_Y(Y_pred)

            # A few metrics
            ring_miscounts, total_obj, pix_err, ipem = calc_errors(Yp, Yv)
            class_acc = (total_obj-ring_miscounts)*1.0/total_obj*100
            acc_hist.append( class_acc )

            # Plot Progress:  Centroids & History
            orig_img_dims=[512,384]
            self.fig = plt.figure(figsize=(14, 3.75))
            self.fig.clf()

            # Plot centroids
            num_plot = 45                           # number of images to plot centroids for
            # for 'noobj' centroids, set the plot color to be fully transparent
            ax = plt.subplot(131, autoscale_on=False, aspect=orig_img_dims[0]*1.0/orig_img_dims[1], xlim=[0,orig_img_dims[0]], ylim=[0,orig_img_dims[1]])

            '''
            for an in range( int(Yv.shape[1]/vars_per_pred)):   # loop over all possible antinodes in array of receptors
                ind = ind_cx + an * vars_per_pred   # index of x position of this antinode
                ind_noobj = ind + 6  # TODO: let's not plot non-objects

                if (0==an):  # this just creates the legend (if you put the label on all of them you get a giant legend, so only call this once)
                    ax.plot(Yv[0:num_plot,ind],Yv[0:num_plot,ind+1],'ro', label="Expected")
                    ax.plot(Yp[0:num_plot,ind],Yp[0:num_plot,ind+1],'go', label="Predicted")
                ax.plot(Yv[0:num_plot,ind],Yv[0:num_plot,ind+1],'ro')
                ax.plot(Yp[0:num_plot,ind],Yp[0:num_plot,ind+1],'go')
            '''
            true_clist_x, true_clist_y = self.make_obj_centroid_list(Yv, num_plot, vars_per_pred)
            pred_clist_x, pred_clist_y = self.make_obj_centroid_list(Yp, num_plot, vars_per_pred)

            ax.plot(true_clist_x, true_clist_y,'ro',label="Expected")
            ax.plot(pred_clist_x, pred_clist_y,'go',label="Predicted")

            ax.set_title('Sample Centroids (cx, cy)')
            ax.legend(loc='upper right', fancybox=True, framealpha=0.8)

            #  Plot history: Loss vs. Time graph
            if not ([] == val_loss_hist):
                ymin = np.min(train_loss_hist + val_loss_hist + center_loss_hist + size_loss_hist + noobj_loss_hist + angle_loss_hist + class_loss_hist) # '+' here concatenates the lists
                ymax = np.max(train_loss_hist + val_loss_hist)
                #TODO: Add loss histories for different *parts* of loss: location, semi's, angle, rings, etc....
                ax = plt.subplot(132, ylim=[np.min((ymin,0.01)),np.min((ymax,0.1))])    # cut the top off at 0.1 if necessary, so we can better see low-error features
                ax.semilogy(hist, train_loss_hist,'-',label="Train")
                ax.semilogy(hist, val_loss_hist,'-',label="Val: Total")
                ax.semilogy(hist, center_loss_hist,'-',label="Val: Center")
                ax.semilogy(hist, size_loss_hist,'-',label="Val: Size")
                ax.semilogy(hist, angle_loss_hist,'-',label="Val: Angle")
                ax.semilogy(hist, noobj_loss_hist,'-',label="Val: NoObj")
                ax.semilogy(hist, class_loss_hist,'-',label="Val: Rings")

                ax.set_xlabel('(Global) Epoch')
                ax.set_ylabel('Loss')
                #ax.set_title('class accuracy = {:5.2f} %'.format(class_acc))
                ax.legend(loc='lower left', fancybox=True, framealpha=0.8)
                plt.xlim(xmin=1)

                # plot accuracy history
                ax = plt.subplot(133, ylim=[0,100])
                ax.plot(hist, acc_hist,'-',color='orange', label='Acc = {:5.2f} %'.format(class_acc))
                ax.set_xlabel('(Global) Epoch')
                ax.set_ylabel('Class Accuracy (%)')
                #if (len(acc_hist) >= 16):
                #    start_at = 4    # ignore everything before a certain epoch
                #    popt, pcov = curve_fit(acc_extrap_func, hist[start_at:], acc_hist[start_at:],p0=[85,0.02,start_at])
                #    print("     Accuracy Extrapolation: ",popt[0],"%")
                #    #ax.plot(hist, acc_extrap_func(hist, *popt), 'c--', label='Fit, Max at {:5.2f} %'.format(popt[0]+popt[1]))
                #    #ax.set_title("Extrapolation:  "+str(popt[0]+popt[1])+"%")
                ax.legend(loc='lower right', fancybox=True, framealpha=0.8)
                #ax.set_title('class accuracy = {:5.2f} %'.format(class_acc))
                plt.xlim(xmin=1)


            self.fig.tight_layout()      # get rid of useless margins
            if not self.use_tb:         # Write image to ordinary file
                plt.savefig(self.log_dir+'/progress.png')
                plt.close(self.fig)
            else:                       # Save image to tensorboard
                plt.savefig(self.buf, format='png')
                self.buf.seek(0)
                image = tf.image.decode_png(self.buf.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)
                summary_op = tf.summary.image("Centroids", image, max_outputs=3, collections=None)
                step = global_count
                summary = self.sess.run(summary_op)
                self.writer.add_summary(summary, step)
                #writer.close()

            show_pred_ellipses(Yv, Yp, val_file_list, log_dir=self.log_dir, ind_extra=ipem)

            # Print additional diagnostics
            print("    In whole val dataset:")
            print('        Mean pixel error =',np.mean(pix_err))
            print("        Max pixel error =",pix_err[ipem]," (index =",ipem,", file=",val_file_list[ipem],").")
            #print("              Y_pred =",Y_pred[ipem])
            #print("              Y_val =",Y_val[ipem])
            print("        Num ring miscounts = ",ring_miscounts,' / ',total_obj,'.   = ',class_acc,' % class. accuracy',sep="")
            #print("                                                                  my_val_loss:",my_val_loss)



class AugmentOnTheFly(Callback):
    """
    This will call some simple data augmentation which does NOT change the 'Y' values
    We do not want to use the Keras ImageDataGenerator class because I don't want 'transformations'

    Examples of OK operations: cutout, noise.
    Not ok:  Anything that would necessitate changing the metadata, e.g. translation, rotation, reflection, scaling

    Inputs: X, the inputs, should be the training dataset
            aug_every,  interval (in epochs) on which to augment
    """
    def __init__(self, X, aug_every=1):
        self.X = X              # this is just a pointer; will get overwritten (this is how we update the training set)
        self.X_orig = X.copy()  # this is an entire copy (Warning: CPU Memory hog!)
        self.aug_every = aug_every


    def salt_n_pepa(self, img, salt_vs_pepper = 0.2, amount=0.004):
        """
        Adds (even more) black & white dots to image
        modified from https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
        """
        push_it = np.random.choice(['good','not good'])   # randomly do it or don't do it
        if push_it != 'good':
            return              # abort and leave image unchanged

        salt_color, pepper_color = np.max(img), np.min(img)
        #print("Salt & Pepa's here, and we're in effect!")
        num_salt = np.ceil(amount * img.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * img.size * (1.0 - salt_vs_pepper))

        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        img[coords[0], coords[1], :] = salt_color

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        img[coords[0], coords[1], :] = pepper_color


    def cutout(self, img, max_regions=6, minsize=11, maxsize=75):
        """
        "Improved Regularization of Convolutional Neural Networks with Cutout", https://arxiv.org/abs/1708.04552
        All we do is chop out rectangular regions from the image, i.e. "masking out contiguous sections of the input"
        Unlike the original cutout paper, we don't cut out anything too huge, and we use random (greyscale) colors
        Note: operates in place on one image
        """
        num_regions = np.random.randint(0, high=max_regions+1)
        if (0 == num_regions):
            return              # abort
        colormin, colormax = np.min(img), np.max(img)
        for region in range(num_regions):
            pt1 = ( np.random.randint(0,img.shape[0]-minsize), np.random.randint(0, img.shape[1]-minsize))  # upper left corner of cutout rectangle
            rwidth, rheight = np.random.randint(minsize, maxsize), np.random.randint(minsize, maxsize)  # width & height of cutout rectangle
            pt2 = ( min(pt1[0] + rwidth, img.shape[0]-1) , min(pt1[1] + rheight, img.shape[1]-1)  )   # keep rectangle bounded in image
            const_value = np.random.uniform( colormin, colormax )
            img[ pt1[0]:pt2[0], pt1[1]:pt2[1], : ] = const_value

    def on_epoch_begin(self, epoch, logs=None):
        # Do the augmentation at the beginning of the epoch
        # TODO: parallelize? currently this runs in serial & hence is probably slow
        if (epoch > 1) and (0 == epoch % self.aug_every):
            for i in range(self.X_orig.shape[0]): # loop over images. TODO: parallelize this
                if (i % 200 == 0):
                    print("   Augmenting on the fly: ",i,"/",self.X_orig.shape[0],"\r",sep="",end="")
                img = self.X_orig[i,:,:,:].copy()  # grab individual image (assume other operations occur in-place)

                # Now do stuff to the image...
                self.cutout(img)
                self.salt_n_pepa(img)

                self.X[i,:,:,:] = img   # overwrite the relevant part of current dataset (this propagates 'out' to model.fit b/c pointers)
            print("")
            
    def on_epoch_end(self, epoch, logs=None):
        pass   # do nothing




def train_network(weights_file="weights.hdf5", datapath="Train/", fraction=1.0):
    np.random.seed(1)

    # Params for training: batch size, ....
     # greater batch size runs faster but may yield Out Of Memory errors
     # also note that small batches yield better generalization: https://arxiv.org/pdf/1609.04836.pdf
    batch_size = 32 #20 for large images, single processor, 40 for dual processor

    print("Loading data, fraction =",fraction)
    print("  Loading Training dataset...")
    X_train, Y_train, img_dims, train_file_list, pred_shape = build_dataset(path=datapath, load_frac=fraction, set_means_ranges=True, batch_size=batch_size)
    valpath="Val/"
    print("  Loading Validation dataset...")
    X_val, Y_val, img_dims, val_file_list, pred_shape  = build_dataset(path=valpath, load_frac=1.0, set_means_ranges=False, batch_size=batch_size)

    print("Instantiating model...")
    parallel=True
    freeze_fac=0.0
    model = setup_model(X_train, Y_train, no_cp_fatal=False, weights_file=weights_file, parallel=parallel, freeze_fac=freeze_fac)

    # Set up callbacks
    #checkpointer = ModelCheckpoint(filepath=weights_file, save_best_only=True)
    checkpointer = ParallelCheckpointCallback(model, filepath=weights_file, save_every=5)

    #history = KerasLossHistory()
    now = time.strftime("%c").replace('  ','_').replace(' ','_')   # date, with no double spaces or spaces
    log_dir='./logs/'+now

    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    patience = 40
    earlystopping = EarlyStopping(patience=patience)
    myprogress = MyProgressCallback(X_val=X_val, Y_val=Y_val, val_file_list=val_file_list, log_dir=log_dir, pred_shape=pred_shape)

    aug_on_fly = AugmentOnTheFly(X_train, aug_every=2)

    callbacks = [myprogress, checkpointer, aug_on_fly, earlystopping, tensorboard]

    frozen_epochs = 0;  # how many epochs to first run with last layers of model frozen
    later_epochs = 400

    # early training with partially-frozen pre-trained model
    if (frozen_epochs > 0) and (freeze_fac > 0.0):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=frozen_epochs, shuffle=True,
                  verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks)
    if (freeze_fac > 0.0):
        model = unfreeze_model(model, X_train, Y_train, parallel=parallel)

    # main training block
    checkpointer = ParallelCheckpointCallback(model, filepath=weights_file, save_every=5)  # necessary b/c unfreezing created new model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=later_epochs, shuffle=True,
              verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks)
    return model



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="trains network on training dataset")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-c', '--datapath', #type=argparse.string,
        help='Train dataset directory with list of classes', default="Train/")
    parser.add_argument('-f', '--fraction', type=float,
        help='Fraction of dataset to use', default=1.0)
    args = parser.parse_args()
    model = train_network(weights_file=args.weights, datapath=args.datapath, fraction=args.fraction)

    # TODO: Score the model against Test dataset
