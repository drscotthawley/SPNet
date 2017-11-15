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
from scipy.optimize import curve_fit


def calc_errors(Yp, Yt):  #index = 2 is where the ring count is stored.
                             # Yp & Yt have already been 'denormalized' at this point
    max_pred_antinodes = int(Yt.shape[1]/vars_per_pred)
    diff = Yp - Yt
    # pixel error in antinode centers
    pix_err = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
    ipem = np.argmax(pix_err)               # index of pixel error maximum

    # number of ring miscounts
    miscounts = 0
    total_obj = 0                           # total is the number of true objects
    for j in range(Yt.shape[0]):
        for an in range(max_pred_antinodes):
            ind = ind_rings + an * vars_per_pred
            rings_t = int(round(Yt[j,ind]))
            i_noobj = ind_noobj + an * vars_per_pred
            if (0 == int(round(Yt[j,i_noobj])) ):   # Is there supposed to be an object there? If so, count the rings
                total_obj += 1
                if (int(round(Yp[j,ind])) != rings_t): # compare integer ring counts
                    miscounts += 1
            elif (int(round(Yt[j,i_noobj])) != int(round(Yp[j,i_noobj]))):  # consider false background as a mistake
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
            Yt = denorm_Y(Y_val)     # t is for true
            Yp = denorm_Y(Y_pred)

            # A few metrics
            ring_miscounts, total_obj, pix_err, ipem = calc_errors(Yp, Yt)
            class_acc = (total_obj-ring_miscounts)*1.0/total_obj*100
            acc_hist.append( class_acc )

            # Plot Progress:  Centroids & History
            orig_img_dims=[512,384]
            self.fig = plt.figure(figsize=(14, 3.75))
            self.fig.clf()

            # Plot centroids
            num_plot = 45                           # number of images to plot centroids for
            ax = plt.subplot(131, autoscale_on=False, aspect=orig_img_dims[0]*1.0/orig_img_dims[1], xlim=[0,orig_img_dims[0]], ylim=[0,orig_img_dims[1]])
            for an in range( int(Yt.shape[1]/vars_per_pred)):
                ind = ind_cx + an * vars_per_pred
                # let's not plot non-objects
                if (0==an):
                    ax.plot(Yt[0:num_plot,ind],Yt[0:num_plot,ind+1],'ro', label="Expected")
                    ax.plot(Yp[0:num_plot,ind],Yp[0:num_plot,ind+1],'go', label="Predicted")
                else:
                    ax.plot(Yt[0:num_plot,ind],Yt[0:num_plot,ind+1],'ro')
                    ax.plot(Yp[0:num_plot,ind],Yp[0:num_plot,ind+1],'go')
            ax.set_title('Sample Centroids (cx, cy)')
            ax.legend(loc='upper right', fancybox=True, framealpha=0.8)

            #  Plot history: Loss vs. Time graph
            if not ([] == val_loss_hist):
                ymin = np.min(train_loss_hist + val_loss_hist + center_loss_hist + size_loss_hist + noobj_loss_hist)
                ymax = np.max(train_loss_hist + val_loss_hist)
                #TODO: Add loss histories for different *parts* of loss: location, semi's, angle, rings, etc....
                ax = plt.subplot(132, ylim=[np.min((ymin,0.01)),np.min((ymax,0.1))])    # cut the top off at 0.1 if necessary, so we can better see low-error features
                ax.semilogy(hist, train_loss_hist,'-',label="Train")
                ax.semilogy(hist, val_loss_hist,'-',label="Val: Total")
                ax.semilogy(hist, center_loss_hist,'-',label="Val: Center")
                ax.semilogy(hist, size_loss_hist,'-',label="Val: Size")
                ax.semilogy(hist, angle_loss_hist,'-',label="Val: Angle")
                ax.semilogy(hist, noobj_loss_hist,'-',label="Val: NoObj")
                ax.semilogy(hist, class_loss_hist,'-',label="Val: Class")

                ax.set_xlabel('(Global) Epoch')
                ax.set_ylabel('Loss')
                #ax.set_title('class accuracy = {:5.2f} %'.format(class_acc))
                ax.legend(loc='upper right', fancybox=True, framealpha=0.8)
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
                ax.legend(loc='bottom right', fancybox=True, framealpha=0.8)
                #ax.set_title('class accuracy = {:5.2f} %'.format(class_acc))
                plt.xlim(xmin=1)


            fig.tight_layout()      # get rid of useless margins
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

            show_pred_ellipses(Yt, Yp, val_file_list, log_dir=self.log_dir, ind_extra=ipem)

            # Print additional diagnostics
            print("    In whole val dataset:")
            print('        Mean pixel error =',np.mean(pix_err))
            print("        Max pixel error =",pix_err[ipem]," (index =",ipem,", file=",val_file_list[ipem],").")
            #print("              Y_pred =",Y_pred[ipem])
            #print("              Y_val =",Y_val[ipem])
            print("        Num ring miscounts = ",ring_miscounts,' / ',total_obj,'.   = ',class_acc,' % class. accuracy',sep="")
            #print("                                                                  my_val_loss:",my_val_loss)



def train_network(weights_file="weights.hdf5", datapath="Train/", fraction=1.0):
    np.random.seed(1)

    # Params for training: batch size, ....
    batch_size = 20 #20 for large images    # greater batch size runs faster but may yield Out Of Memory errors
                                            # also note that small batches yield better generalization: https://arxiv.org/pdf/1609.04836.pdf

    print("Getting data..., fraction = ",fraction)
    X_train, Y_train, img_dims, train_file_list, pred_shape = build_dataset(path=datapath, load_frac=fraction, set_means_ranges=True, batch_size=batch_size)
#    testpath="Test/"
#    X_test, Y_test, img_dims, test_file_list  = build_dataset(path=testpath, load_frac=fraction)
    valpath="Val/"
    X_val, Y_val, img_dims, val_file_list, pred_shape  = build_dataset(path=valpath, load_frac=fraction, set_means_ranges=False, batch_size=batch_size)

    print("Instantiating model...")
    parallel=True
    model = setup_model(X_train, Y_train, no_cp_fatal=False, weights_file=weights_file, parallel=parallel)

    # Set up callbacks
    checkpointer = ModelCheckpoint(filepath=weights_file, save_best_only=True)
    #history = KerasLossHistory()
    now = time.strftime("%c").replace('  ','_').replace(' ','_')   # date, with no double spaces or spaces
    log_dir='./logs/'+now

    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    patience = 40
    earlystopping = EarlyStopping(patience=patience)
    myprogress = MyProgressCallback(X_val=X_val, Y_val=Y_val, val_file_list=val_file_list, log_dir=log_dir, pred_shape=pred_shape)

    frozen_epochs = 0;  #MobileNet is robust enough that I don't need to pre-train my final layers first
    later_epochs = 400

    # early training with partially-frozen pre-trained model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=frozen_epochs, shuffle=True,
              verbose=1, validation_data=(X_val, Y_val), callbacks=[checkpointer,earlystopping,tensorboard,myprogress])
    model = unfreeze_model(model, X_train, Y_train, parallel=parallel)

    # main training block
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=later_epochs, shuffle=True,
              verbose=1, validation_data=(X_val, Y_val), callbacks=[checkpointer,earlystopping,tensorboard,myprogress])

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

    # Score the model against Test dataset
