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

---
Alternate readme... need to merge this with above...


# Machine-Learning Labeling of [ESPI](https://en.wikipedia.org/wiki/Electronic_speckle_pattern_interferometry) Data

_S.H. Hawley, Oct 2017_

**Goal**: Assist with the [Steelpan Vibrations](https://www.zooniverse.org/projects/achmorrison/steelpan-vibrations) project, using machine learning trained on human-labeled data

**Algorithm**: This falls under "[object detection](https://en.wikipedia.org/wiki/Object_detection)". Uses a convolutional neural network outputting bounding ellipses and ring counts. Specificially, the CNN is either [MobileNet](https://arxiv.org/abs/1704.04861) or <a href="">Inception-ResnetV2</a> (can switch them to [trade off speed vs. accuracy](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_SpeedAccuracy_Trade-Offs_for_CVPR_2017_paper.pdf)), and prediction scheme is a modification of [YOLO9000](https://arxiv.org/abs/1612.08242) to predict rotated ellipses, and to predict number-of-rings count via regression (just easier to code; TODO: switch to classification)

**Code**: [SPNet](https://github.com/drscotthawley/SPNet) (private repo)

**Strategy**: Train on 'fake' images to test algorithm, switch to [@achmorrison](https://twitter.com/achmorrison)'s 'real' data at some point.

**Non-machine-learning analogue**: [Elliptical Hough Transform](http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html). The EHT reportedly doesn't scale well, or handle noise well, whereas the neural network does both.

