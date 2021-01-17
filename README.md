# SPNet - Object detection for [ESPI](https://en.wikipedia.org/wiki/Electronic_speckle_pattern_interferometry)  images of oscillating steelpan drums

_S.H. Hawley, Oct 2017-Jan 2020._

_Warning: This is "research" code, modified many times over the span of 3+ years to support only one user (me). It is shared publicly here for the purposes of transparency and verification,  but it should not be regarded as a "package" or library maintained for general public use._



Sample image:

![sample image](http://hedges.belmont.edu/~shawley/steelpan/steelpan_pred_00002.png)

**Goal**: Assist with the [Steelpan Vibrations](https://www.zooniverse.org/projects/achmorrison/steelpan-vibrations) project, using machine learning trained on human-labeled data

**Algorithm**: This falls under "[object detection](https://en.wikipedia.org/wiki/Object_detection)". Uses a convolutional neural network outputting bounding ellipses and ring counts. Specificially, the CNN is either [MobileNet](https://arxiv.org/abs/1704.04861) or <a href="">Inception-ResnetV2</a> (can switch them to [trade off speed vs. accuracy](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_SpeedAccuracy_Trade-Offs_for_CVPR_2017_paper.pdf)), and prediction scheme is a modification of [YOLO9000](https://arxiv.org/abs/1612.08242) to predict rotated ellipses, and to predict number-of-rings count via regression (just easier to code; TODO: switch to classification)

**Code**: [SPNet](https://github.com/drscotthawley/SPNet) (private repo)

**Strategy**: Train on 'fake' images to test algorithm, switch to [@achmorrison](https://twitter.com/achmorrison)'s 'real' data at some point.

**Non-machine-learning analogue**: [Elliptical Hough Transform](http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html). The EHT reportedly doesn't scale well, or handle noise well, whereas the neural network does both.


Uses a [YOLO](https://pjreddie.com/darknet/yolo/)-style approach, but fits ellipses instead of boxes, and performs regression instead of classification -- counts the number of rings.

Built in [Keras](https://keras.io/), running [MobileNet](https://arxiv.org/abs/1704.04861) for image classification.



## Minimal usage documentation:

### Installation
Create a conda environment, but use pip for package installs
```
git clone git@github.com:drscotthawley/SPNet.git
cd SPNet
conda create -y --name spnet python=3.5.4
conda activate spnet
pip install -r requirements.txt
```
(To remove the environment: `conda env remove --name spnet`)

### Syntethic Data:
The 'real' datasets is Andrew Morrison's business. But you can test SPNet using 'fake' images:
    ./gen_fake_espi
Generates 50,000 fake images, placing them in directories Train, Val and Test.


    ./train_fake_espi
Actually does the training.  It has a few options, e.g. where files are/go, and 
how much of dataset to use.  Try running `./train_fake_espi --help`

More later.

### 'Real' Data:

#### Workflow:
The following assumes SPNet/ is in your home directory, and you're on a Unix-like system.

*Hawley note to self: run `source activate py36` on lecun to get the correct environment*

1. Obtain single .csv file of (averaged) Zooniverse output (e.g. from achmorrison), and rename it `zooniverse_labeled_dataset.csv` (TODO: offer command line param for filename)
2. From the directory where `zooniverse_labeled_dataset.csv` resides, place all relevant images in a sub-directory `zooniverse_steelpan/`
3. From within same directory as `zooniverse_labeled_dataset.csv`, run the `parse_zooniverse_csv.py` utility, e.g. run `cd ~/datasets; ~/SPNet/parse_zooniverse_csv.py`.   This will place both images and new .csv files in a new directory called  `parsed_zooniverze_steelpan/`.  
4. As a check, list what's in the output directory: `ls parsed_zooniverze_steelpan/`
5. As a check, try editing these images, e.g. ` ~/SPNet/ellipse_editor.py parsed_zooniverze_steelpan`  (no slash on the end)
6. Now switch to the SPNet/ directory: `cd ~/SPNet`
7. "Set Up" the Data: Run `./setup_data.py`.  This will segment the dataset into Train, Val & Test subsets,
*and* do augmentation on (only) the Train/ data.  (If later you want to re-augment, you can run `augment_data.py` alone.)   Note:The augmentation will also include synthetic data.
u. Now you should be ready to train: ` ~/SPNet/train_spnet.py `

## Making a movie
`./predict_network.py` will output a list of `.png` files in `logs/Predicting`.  To turn them into an mp4 movie named `out.mp4`, cd in to the `logs/Predicting` directory and then run

```bash
ffmpeg -r 1/5 -i steelpan_pred_%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4
```

## Cite as:
```
@article{spnet_hawley_morrison,
  author={Scott H. Hawley and Andrew C. Morrison},
  title={ConvNets for Counting: Object Detection of Antinode Displacements in Oscillating Steelpan Drums},
  url={arXiv:??},
  month={Jan},
  year={2021},
  journal={Submitted to Special Issue on Machine Learning in Acoustics, Journal of the Acoustical Society of America (JASA)},
}
```

### Related:
Slides from talk at Dec. 2019 Acoustical Society meeting: [https://hedges.belmont.edu/~shawley/SPNET_ASA2019.pdf](https://hedges.belmont.edu/~shawley/SPNET_ASA2019.pdf)

--
Scott H. Hawley
