# SPNet - Object detection for ESPI images of oscillating steelpan drums

Designed to help with the steelpan-oscillation problem (https://www.zooniverse.org/projects/achmorrison/steelpan-vibrations)

Sample image:

![sample image](http://hedges.belmont.edu/~shawley/steelpan/steelpan_pred_00002.png)

Uses a [YOLO](https://pjreddie.com/darknet/yolo/)-style approach, but fits ellipses instead of boxes, and performs regression instead of classification -- counts the number of rings.

Built in [Keras](https://keras.io/), running [MobileNet](https://arxiv.org/abs/1704.04861) for image classification.

Author: Scott H. Hawley

Minimal usage documentation:

    ./gen_fake_espi
Generates 50,000 fake images, placing them in directories Train, Val and Test.

    ./train_fake_espi
Actually does the training.  Has a few options, e.g. where files are/go, and how much of dataset to use.  Try --help.

More later.
