import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import glob
from keras.callbacks import Callback
import cv2
import keras.backend as K
import tensorflow as tf
from spnet import models, diagnostics
from spnet.config import *
from spnet.utils import *
from spnet.augmentation import *
import io
from numba import jit

# Custom  Keras callbacks

class ParallelCheckpointCallback(Callback):
    """
    UPDATE: I (SHH) recommend NOT running old-style Keras in parallel AT ALL;
    thus this would not get called.

    Keras & HDF5 don't play nice when checkpointing data-parallel models,
    so we use the 'serial part' as per @fchollet's remarks in
    https://github.com/keras-team/keras/issues/8649#issuecomment-348829198
    """
    def __init__(self, serial_model, filepath="weights.hdf5", save_every=1, dir='.'):
         self.model_to_save = serial_model
         self.save_every = save_every
         self.weights_path = dir+'/'+filepath
         self.model_path = dir+'/'+"spnet.model"

    def on_epoch_end(self, epoch, logs=None):
        # Note the epoch+1 below is to agree w/ Keras' display of epoch+1 when it writes to the screen, e.g. "Epoch 1/20"
        if (1 == self.save_every) or ( (0 == ((epoch+1) % self.save_every)) and (epoch > 0) ):
            print("Saving weights checkpoint to",self.weights_path)
            self.model_to_save.save_weights(self.weights_path)  # on restart, we only load the weights
            print("Saving entire model checkpoint to",self.model_path)
            self.model_to_save.save(self.model_path)          # but also save the whole model


# for tracking progress
hist = []
train_loss_hist = []
val_loss_hist = []
my_val_loss_hist = []
acc_hist = []
center_loss_hist = []
size_loss_hist = []
angle_loss_hist = []
noobj_loss_hist = []
rings_loss_hist = []

global_count = 0                	       # global_count is handy when nb_epoch = 1
n_epochs_per_plot = 1
class MyProgressCallback(Callback):      # Callbacks essentially get inserted into code at model.fit()
    """
    This is a callback routine for visualizing progress of training and
    printing various measurments and diagnostics
    """
    def __init__(self, X_val=None, Y_val=None, val_file_list=None,
        log_dir="./logs", use_tb=False, pred_shape=[3,3,4,6]):
        self.X_val = X_val
        self.Y_val = Y_val
        self.val_file_list = val_file_list
        self.buf = io.BytesIO()
        self.log_dir = log_dir
        self.use_tb = use_tb              # use tensorboard
        #self.max_pred_antinodes=max_pred_antinodes
        self.pred_shape = pred_shape

        if not os.path.exists(log_dir):  # make sure log directory exists
            os.makedirs(log_dir)

        self.loss_file = open(log_dir+'/losses.dat','a')
        self.loss_file.write(f'# epoch Train_total Val_total center size angle noobj class\n')


    def on_train_begin(self, logs={}):
        hist = []
        train_loss_hist = []
        val_loss_hist = []
        my_val_loss_hist = []
        acc_hist = []

        if (self.use_tb):   # Using tensorboard
            self.sess = K.get_session()
            self.writer = tf.summary.FileWriter(self.log_dir)

    def make_obj_centroid_list(self, Y, num_centroids):
        """
        This constructs a list of centroids of actual objects (noobj==0), up to a certain limit (num_centroids)
        Y: a grid of predictors, as in either Yv or Yp.  see on_epoch_end(), below
        num_centroids: The number of centroids to try to grab for plotting below
        """
        print('    Making list of real centroids...')
        xlist, ylist = [], []
        row, maxrow = 0, Y.shape[0]-1
        while (len(xlist) < num_centroids) and (row < maxrow):
            # our predictors are arranged as multiple groups of columns (cf.vars_per_pred in each group)
            for an in range( int(Y.shape[1]/cf.vars_per_pred)):   # loop over all possible antinodes in array of receptors
                ind = cf.ind_cx + an * cf.vars_per_pred   # index of x position of this antinode
                ind_noobj = ind + 6
                if (0==int(round(Y[row,ind_noobj]))):   # if object exists
                    xlist.append(Y[row,ind])
                    ylist.append(Y[row,ind+1])
            row += 1
        return xlist, ylist


    def on_epoch_end(self, epoch, logs={}):
        """
        Here's where most of the diagnostics & viz happens
        This is really long! ;-)
        """
        global global_count                 # we use global_count instead of epoch # b/c of "Outer" loop
        global_count = global_count+1

        #self.count = self.count + 1
        if (0 == global_count % n_epochs_per_plot):  # only do this stuff every n_epochs_per_plot
            hist.append(global_count)
            train_loss_total, val_loss_total = logs.get('loss'), logs.get('val_loss')
            train_loss_hist.append(train_loss_total)
            val_loss_hist.append(val_loss_total)

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
            my_val_loss, loss_parts = models.my_loss(Y_val, Y_pred)
            [center_loss, size_loss, angle_loss, noobj_loss, rings_loss] = loss_parts
            my_val_loss_hist.append(my_val_loss)
            center_loss_hist.append(center_loss)
            size_loss_hist.append(size_loss)
            angle_loss_hist.append(angle_loss)
            noobj_loss_hist.append(noobj_loss)
            rings_loss_hist.append(rings_loss)

            loss_dat_str = f'{epoch} {train_loss_total} {my_val_loss} {center_loss} {size_loss} {angle_loss} {noobj_loss} {rings_loss}\n'
            self.loss_file.write(loss_dat_str)

            if cf.loss_type != 'same':  # convert from logits
                Y_pred[:, cf.ind_noobj::cf.vars_per_pred] = 1.0/(1.0 + np.exp(-Y_pred[:, cf.ind_noobj::cf.vars_per_pred])) # sigmoid

            # calc some errors.  first transform back into regular 'world' values via de-normalization
            Yv = denorm_Y(Y_val)     # t is for true
            Yp = denorm_Y(Y_pred)

            # A few metrics
            ring_miscounts, ring_truecounts, total_obj, false_obj_pos, false_obj_neg, true_obj_pos, true_obj_neg, pix_err, ipem = diagnostics.calc_errors(Yp, Yv)
            mistakes = ring_miscounts + false_obj_pos + false_obj_neg
            class_acc = (total_obj-mistakes)*1.0/total_obj*100  # accuracy is based on lack of any mistakes
            #print("Diagnostics: ring_miscounts, total_(true)_obj, false_obj_pos, false_obj_neg, true_obj_pos, true_obj_neg, ipem = \n",
            #    "             ",ring_miscounts, total_obj, false_obj_pos, false_obj_neg, true_obj_pos, true_obj_neg, ipem) # skip pix_err as it's an array
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
            for an in range( int(Yv.shape[1]/cf.vars_per_pred)):   # loop over all possible antinodes in array of receptors
                ind = ind_cx + an * cf.vars_per_pred   # index of x position of this antinode
                ind_noobj = ind + 6  # TODO: let's not plot non-objects

                if (0==an):  # this just creates the legend (if you put the label on all of them you get a giant legend, so only call this once)
                    ax.plot(Yv[0:num_plot,ind],Yv[0:num_plot,ind+1],'ro', label="Expected")
                    ax.plot(Yp[0:num_plot,ind],Yp[0:num_plot,ind+1],'go', label="Predicted")
                ax.plot(Yv[0:num_plot,ind],Yv[0:num_plot,ind+1],'ro')
                ax.plot(Yp[0:num_plot,ind],Yp[0:num_plot,ind+1],'go')
            '''
            true_clist_x, true_clist_y = self.make_obj_centroid_list(Yv, num_plot)
            pred_clist_x, pred_clist_y = self.make_obj_centroid_list(Yp, num_plot)

            ax.plot(true_clist_x, true_clist_y,'ro',label="Expected")
            ax.plot(pred_clist_x, pred_clist_y,'go',label="Predicted")

            ax.set_title('Sample Centroids (cx, cy)')
            ax.legend(loc='upper right', fancybox=True, framealpha=0.8)

            #  Plot history: Loss vs. Time graph
            if not ([] == val_loss_hist):
                ymin = np.min(train_loss_hist + val_loss_hist + center_loss_hist + size_loss_hist
                    + noobj_loss_hist + angle_loss_hist + rings_loss_hist) # '+' here concatenates the lists
                print("ymin = ",ymin)
                ymax = np.max(train_loss_hist + val_loss_hist)
                ax = plt.subplot(132, ylim=[5e-6, 0.1])    # cut the top off at 0.1 if necessary, so we can better see low-error features

                print('    Losses: Epoch Train:total Val:total   center     size       angle      noobj      rings')
                print(f'          {epoch:3d}     {train_loss_total:.3e}   {val_loss_total:.3e}   {center_loss:.3e} '+
                    f' {size_loss:.3e}  {angle_loss:.3e}  {noobj_loss:.3e}  {rings_loss:.3e}')
                ax.loglog(hist, train_loss_hist,'-',label="Train")
                ax.loglog(hist, val_loss_hist,'-',label="Val: Total")
                ax.loglog(hist, center_loss_hist,'-',label="Val: Center")
                ax.loglog(hist, size_loss_hist,'-',label="Val: Size")
                ax.loglog(hist, angle_loss_hist,'-',label="Val: Angle")
                ax.loglog(hist, noobj_loss_hist,'-',label="Val: NoObj")
                ax.loglog(hist, rings_loss_hist,'-',label="Val: Rings")

                ax.set_xlabel('(Global) Epoch')
                ax.set_ylabel('Loss')
                #ax.set_title('class accuracy = {:5.2f} %'.format(class_acc))
                ax.legend(loc='lower left', fancybox=True, framealpha=0.8)
                plt.xlim(xmin=1)

                # plot accuracy history
                ax = plt.subplot(133, ylim=[0,100])
                ax.plot(hist, acc_hist,'-',color='orange', label='Acc = {:5.2f} %'.format(class_acc))
                ax.set_xlabel('(Global) Epoch')
                ax.set_ylabel('Accuracy (%)')
                ax.legend(loc='lower right', fancybox=True, framealpha=0.8)
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
            print("    Ring correct counts = ",ring_truecounts,' / ',total_obj,'.   = ',100*ring_truecounts/total_obj,' % ring-class accuracy',sep="")

            print("         Ring miscounts = ",ring_miscounts,' / ',total_obj,'.   = ',100*ring_miscounts/total_obj,' % ring-miscount rate',sep="")
            print("        False positives = ",false_obj_pos,' / ',total_obj,'.   = ',100*false_obj_pos/total_obj,' % FP rate',sep="")
            print("        False negatives = ",false_obj_neg,' / ',total_obj,'.   = ',100*false_obj_neg/total_obj,' % FN rate',sep="")
            print("         True positives = ",true_obj_pos,' / ',total_obj,'.   = ',100*true_obj_pos/total_obj,' % TP rate',sep="")
            print("         True negatives = ",true_obj_neg,sep="")
            print("    Total Mistakes = ",mistakes,' / ',total_obj,'.   => ',class_acc,' % class. accuracy rate (lack of mistakes)',sep="")

            #print("                                                                  my_val_loss:",my_val_loss)



# This is called during training via Keras callback
class AugmentOnTheFly(Callback):
    """
    Keras Callback, called from within model.fit
    This will call some simple data augmentation which does NOT change the annotation 'Y' values
        Because by the time we get to fit, the Y values have been distributed to
        the 'grid' and zero-meaned & unit-varianced via utils.true_to_pred_grid()
    We do not want to use the Keras ImageDataGenerator class because I want more control

    Examples of OK operations: cutout, noise.
    Not ok:  Anything that would depend on the scaling or cropping:  e.g. translation, scaling, shear

    Inputs: X, the inputs, should be the training dataset
            Y, the target metadata
            aug_every,  interval (in epochs) on which to augment
    Note that in augment_data.py, the metadata is different -- it uses angle instead of cosine & sine, and has no "noobj" field
    """
    def __init__(self, X, Y, orig_img_shape=(384,512), aug_every=1):
        self.X = X              # this is just a pointer; will get overwritten (this is how we update the training set)
        self.Y = Y
        self.X_orig = X.copy()  # this is an entire copy (Warning: CPU Memory hog!)
        self.Y_orig = Y.copy()
        self.aug_every = aug_every
        self.orig_img_shape = orig_img_shape

    def salt_n_pepa(self, img, salt_vs_pepper = 0.2, amount=0.004): # IN PLACE
        # moved most of the code to augmentation.py
        salt_n_pepa_inplace(img, salt_vs_pepper=salt_vs_pepper, amount=amount)
        return

    def cutout(self, img, max_regions=6, minsize=11, maxsize=75): # IN PLACE
        # moved most of the code to augmentation.py
        cutout_inplace(img, max_regions=max_regions, minsize=minsize, maxsize=maxsize)
        return

    def blur(self, img, blur_prob=0.4):
        if np.random.rand() < blur_prob:
            blur_inplace(img)
        return

    def bp_mixup(self, img, bpmix_prob=0.3):
        # unlike some other augmentation methods, this is not done in place,
        # and the spectral computations can be slow compared to simply adding noise, etc.
        if np.random.rand() < bpmix_prob:
            img = bandpass_mixup(img)
        return img

    @jit() # jit makes it fast
    def on_epoch_begin(self, epoch, logs=None):
        # Do the augmentation at the beginning of the epoch
        # Numba jit is actually faster than my 'parallel' implementation. Ignore numba warnings btw
        if (0 == epoch % self.aug_every):
            for i in range(self.X_orig.shape[0]): # loop over images. TODO: parallelize this
                if ((i % 200 == 0) or (i == self.X_orig.shape[0]-1)):
                    print("   Augmenting on the fly: ",i+1,"/",self.X_orig.shape[0],"\r",sep="",end="")
                img = self.X_orig[i,:,:,:].copy()  # grab individual image (assume other operations occur in-place)
                metadata = self.Y_orig[i,:].copy()

                # Now do stuff to the image copy
                self.cutout(img)
                self.salt_n_pepa(img)
                self.blur(img)
                #img = self.bp_mixup(img) # this is not in-place


                self.X[i,:,:,:] = img   # overwrite the relevant part of current dataset (this propagates 'out' to model.fit b/c pointers)
                self.Y[i,:] = metadata
            print("")

    def on_epoch_end(self, epoch, logs=None):
        pass   # do nothing



# This goes with the LR schedule callback, below
def get_1cycle_schedule(lr_max=1e-3, n_data_points=8000, epochs=200, batch_size=40, verbose=0):
  """
  Creates a look-up table of learning rates for 1cycle schedule with cosine annealing
  See @sgugger's & @jeremyhoward's code in fastai library: https://github.com/fastai/fastai/blob/master/fastai/train.py
  Wrote this to use with my Keras and (non-fastai-)PyTorch codes.
  Note that in Keras, the LearningRateScheduler callback (https://keras.io/callbacks/#learningratescheduler) only operates once per epoch, not per batch
      So see below for Keras callback

  Keyword arguments:
    lr_max            chosen by user after lr_finder
    n_data_points     data points per epoch (e.g. size of training set)
    epochs            number of epochs
    batch_size        batch size
  Output:
    lrs               look-up table of LR's, with length equal to total # of iterations
  Then you can use this in your PyTorch code by counting iteration number and setting
          optimizer.param_groups[0]['lr'] = lrs[iter_count]
  """
  if verbose > 0:
      print("Setting up 1Cycle LR schedule...")
  pct_start, div_factor = 0.3, 25.        # @sgugger's parameters in fastai code
  lr_start = lr_max/div_factor
  lr_end = lr_start/1e4
  n_iter = n_data_points * epochs // batch_size     # number of iterations
  a1 = int(n_iter * pct_start)
  a2 = n_iter - a1

  # make look-up table
  lrs_first = np.linspace(lr_start, lr_max, a1)            # linear growth
  lrs_second = (lr_max-lr_end)*(1+np.cos(np.linspace(0,np.pi,a2)))/2 + lr_end  # cosine annealing
  lrs = np.concatenate((lrs_first, lrs_second))
  return lrs


class OneCycleScheduler(Callback):
    """My modification of Keras' Learning rate scheduler to do 1Cycle learning
       which increments per BATCH, not per epoch
    Keyword arguments
        **kwargs:  keyword arguments to pass to get_1cycle_schedule()
        Also, verbose: int. 0: quiet, 1: update messages on epoch, 2: update after batches

    Sample usage (from my train.py):
        lrsched = OneCycleScheduler(lr_max=1e-4, n_data_points=X_train.shape[0], epochs=epochs, batch_size=batch_size, verbose=1)
    """
    def __init__(self, **kwargs):
        super(OneCycleScheduler, self).__init__()
        self.verbose = kwargs.get('verbose', 0)
        self.lrs = get_1cycle_schedule(**kwargs) # get a look-up table of learing rate values per iteration
        self.iteration = 0

    def on_batch_begin(self, batch, logs=None):
        # assign the learning rate for this batch by reading from look-up table
        K.set_value(self.model.optimizer.lr, self.lrs[self.iteration])
        self.iteration += 1

    def on_epoch_end(self, epoch, logs=None):       # this is unchanged from Keras LearningRateScheduler
        logs = logs or {}
        lr = K.get_value(self.model.optimizer.lr)
        logs['lr'] =lr
        if self.verbose > 0:
            print('\nLearning rate =',lr)
