[//]: # (Image References)

[image1]: ./examples/kitti.png
[image2]: ./examples/gti_far.png
[image3]: ./examples/gti_middle.png
[image4]: ./examples/gti_left.png
[image5]: ./examples/gti_right.png
[image6]: ./examples/extra_non0.png
[image7]: ./examples/extra_non1.png
[image8]: ./examples/extra_non2.png
[image9]: ./examples/gti_non0.png
[image10]: ./examples/gti_non1.png
[image11]: ./examples/resize.jpg
[image12]: ./examples/normal.jpg
[image13]: ./examples/luv.jpg
[image14]: ./examples/20x20.jpg
[image15]: ./examples/histograms.png
[image16]: ./examples/hog.png
[image16a]: ./examples/hog_noncar.png
[image17]: ./test_images/test4.jpg
[image18]: ./examples/img_roi64.jpg
[image19]: ./examples/img_roi160.jpg
[image20]: ./examples/detected.jpg
[image21]: ./examples/heatmap.jpg
[image22]: ./examples/thresholded.jpg
[image23]: ./examples/label_img.jpg
[image24]: ./examples/original.jpg
[image25]: ./examples/near.jpg
[image26]: ./examples/cropped.jpg
[image27]: ./output_images/test4.jpg


# Vehicle Detection
The Project
---


**Vehicle Detection Project**

The goals/steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
* Estimate the distance to detected vehicles
* Estimate relative speed of the detected vehicles

**Running the code**
First, you need to unzip the `vehicles.zip` and `non-vehicles.zip` from the folder `train_images` in the same folder. Next, to train the classifier run python script `train_net.py` which will create the `classifier.p` a pickle dump that contains the trained classifier. After that, you are good to run the `car_finder.py` which looks for cars in a video and estimates distances to them and also looks for lanes. 


---

## The Car Classifier

First thing that we need to do is to create a classifier which classifies car agains non-cars. To do so the dataset is neede, and I have used the dataset provided by Udacity. ( Download: [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip), [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)). The dataset is a combination of KITTI vision benchmark suite and GTI vehicle image database. GTI car images are grouped into far, left, right, middle close. The examples of cars and non cars follow:

| KITTI     | GTI Far     | GTI  Near|    GTI Left | GTI Right  |
|-----------|-------------|------------|-------------|------------|
|![][image1]| ![][image2] | ![][image3]| ![][image4] |![][image5] |
|  Non car  |  Non car    | Non Car    | GTI Non 1   | GTI Non 2  |
|![][image6]| ![][image7] | ![][image8]| ![][image9] |![][image10]|

To build a classifier, first, the features have to be identified. The features that are going to be used is a mixture of histograms, full images, and HOG-s. The calculation of the features is implemented in functions `CarFinger.get_features()` and `CarFinder.car_find_roi()`. The first one is used for training a classifier while the later one is used to batch classify the images. 

### Extracting features
#### Color space
Color space is related to the representation of images in the sense of color encodings. There are encodings that are more suitable for one purpose but bad for the others. For example, RBG is good from the hardware standpoint of view, since it is how the pixels are captured and displayed ([Bayer filter](https://en.wikipedia.org/wiki/Bayer_filter) is one good example) but it does not capture the way humans perceive the colors, which is important for classification tasks. For the task of classifying cars, I am sure that there is no prescribed color space which works the best. So it has to be chosen by trial and error. What I have done, is that I have built the classifier, based on HOG, color histograms, and full image and then changed the color space until I got the best classification result on a test set. Maybe I am describing problem from top to bottom, but the color space is quite important for explaining and visualizing features. After some trial, I found that [LUV](https://en.wikipedia.org/wiki/CIELUV) color space works the best. It has the luminescence component L, as well as two (*u* and *v*) chromaticity components. That color space consistently gave better classification results. The conversion takes place in lines 155 and 182 of file `car_finder.py`.

#### Subsampled and normalized image as a feature
The first and most simple feature is the subsampled image. Once again, by trying and checking classification result the size of a subsampled image is chosen to be 20x20. Also, the image is gamma-normalised. That came as an idea while looking to this [YouTube](https://www.youtube.com/watch?v=7S5qXET179I) video explaining HOG. It was stated that taking a square root of the image normalizes it and gets uniform brightness thus reducing the effect of shadows. I gave it a try and it creates a minute improvement on the classification. Since it is quite a simple operation it stayed in my code since it provides additional robustness. The subsampling and normalization take place in lines 181-183 and 154-156 of file `car_finder.py`. After normalization and subsampling the image is reshaped into a single vector using `numpy.ravel()` The original image normalized converted to LUV and subsampled are:

| Original  | Normalized    | LUV         |Subsampled   |
|-----------|---------------|-------------|-------------|
|![][image11]| ![][image12] | ![][image13]|![][image14] |

#### Histogram of colors
The second group of features is color histograms. Initially, I have tried to use only the histogram of the luminescence channel *L*. The cars can have different colors, so omitting the chromaticity channels looked like a natural choice to me. After some testing, I found out that including a histogram of all three color channels improved the test accuracy for a couple of percents which can make a lot of difference. The histogram calculation is performed in lines 157-159 and 207-210 of `car_finder.py`. A number of bins in a histogram is selected based on the testing accuracy and 128 bins produce the best result. Here are samples of histograms for image previously shown. 

![][image15]

#### HOG
The last, but probably the most important feature is a histogram of oriented gradients. The image on which the HOG is calculated is of size 64x64. The number of pixels per cell is 8, while the number of cells per block is 1. The number of orientations is 12. The HOG is calculated on all three channels of a normalized image. I have tested these parameters a lot and finally found that this was the optimal choice. What I have considered in selection was the number of features generated this way and the obtained accuracy on a test set. When this set of parameters were used,  a total of 768 features per channel are created. If the number of cells per block is increased to 2, the number of features blows up to 2352 per channel. Increase in classification accuracy when using 2 cells per block wasn't substantial so I have chosen to use 1 cell per block. Also, I have tried a higher number of pixels per cell, in what case lot of information is lost and accuracy drops while lowering the number of pixels per cell increases the number of features. Calculation of HOGs is implemented in lines 160-168 and 184-192 of `car_finder.py` Images visualizing HOG for each channel are:

|HOG of a car|
|------------|
|![][image16]|

|HOG of non-car|
|--------------|
|![][image16a] |

#### Training the classifier

The classifier used is *linear support vector classifier* implemented as part of `scikit-learn`. The dataset is obtained by looping through images of vehicles and non-vehicles and calculating features for those images. Next thing that was performed is to scale the features, which is quite important for any machine learning algorithm. After that, the dataset is split into a training and test set, where the test set is 10% of all the data. The classifier is trained with *C=1e-4*, where this feature was selected based on the accuracies of train and test set. If the difference between the two accuracies is high the training is overfitting the data so the *C* was lowered. When the test accuracy is low but same as training accuracy, the underfitting has occurred so the value of the *C* was increased. The final accuracy obtained on the test set was **99.55%**. After the training, the classifier and scaler were pickled and saved, so that they can be reused when images coming from the camera were processed. The implementation of whole training procedure is in file `train_svm.py`. 

---

## Finding cars on images/videos

The pipeline for finding cars in images and videos is very similar. In fact, the finding cars in videos follow the same pipeline for finding cars in still images with some additional features. For that reason, the pipeline for a single image will be described first.

### Sliding the window

The first thing to do is to slide the window across the screen and try to identify the areas which produce a positive hit at defined classifier. The window that is going to get slid is always the size 64x64 with the overlap of 75%. In some cases the car can be bigger than 64x64 pixels so to encompass those cases, the whole image is downscaled. As a result, the car is searched on original, and 5 downscaled images, selected so that the cars on the original image would be of sizes 80x80, 96x96, 112x112, 128x128 and 160x160. HOG is calculated only once per downscaled image and the subregion of HOG is used when each of windows gets tested if there is a car. After the window is slid the whole batch of features calculated for each window gets classified. Windows classified as non-cars get discarded. The whole procedure is implemented in `CarFinder.car_find_roi()`. Here is the example of an original image, 2 regions that get searched for cars and regions with detected cars:

| Original   | Search 64x64   |
|------------|----------------|
|![][image17]| ![][image18]   |
| Detections |  Search 160x160| 
|![][image20]| ![][image19]   |

### Calculating the heatmap and identifying cars

Since there are multiple detections of the same car the windows had to be grouped somehow. For that case, the heatmap is used. Each pixel in the heatmap holds the number of windows with identified cars which contain that pixel. The higher the value of the pixel in the heatmap the more likely it is the part of the car. The heatmap is thresholded with a threshold of 1, which removes any possible false car detections. After that, the connected components get labeled and the bounding box is calculated. This is performed in lines 238 - 254 of `car_finder.py`. The resulting images are:

| Heatmap   | Thresholded Heatmap | Labels      |
|-----------|---------------------|-------------|
|![][image21]| ![][image22]       | ![][image23]|


### Removal of the road from the bottom side of the bounding box. 

The reason for this is that we want to estimate how far ahead of us is identified car. In our previous [project](https://github.com/ajsmilutin/CarND-Advanced-Lane-Lines/blob/master/README.md) the perspective transform was found which maps the road surface to the image and enables us to measure distances. Perspective transform assumes that object transformed is planar, so to measure distance accurately we need a point which is on the road surface. To measure the distance the midpoint of the lower edge of a bounding box is used. The road surface is removed so that the measurement is performed to the back wheels of the identified car. To do so, the median color of the last 8 lines of the bounding box is found. The first line from the bottom in which more than  20% of the points are far from the median color is regarded as the new bottom edge of the bounding box. Points 'far' in color are represented in purple color in the figure below. This procedure is implemented in lines 257-264 of `car_finder.py`

| Original   | Points near in color | Cropped     |
|------------|----------------------|-------------|
|![][image24]| ![][image25]         | ![][image26]|


### Estimating the distance
Before the rectangles around the detected cars are drawn, the lane line is identified. Also, we'll try to assess the distance to the car. Once we get the bounding box and midpoint of its bottom edge, using perspective transform, we can calculate its position on the warped image from the [Advanced lane finding project](https://github.com/ajsmilutin/CarND-Advanced-Lane-Lines/blob/master/README.md). On that image, there is a direct correlation between the pixel position and distance in meters, so the distance between the calculated position of the midpoint and the bottom of the image represents the distance between our car and the car we have detected. By looking how that distance changes from frame to frame, we can calculate car's relative speed, by multiplying the difference between two frames by frames per second and 3.6 to convert it to kilometers per hour, instead of meters per second. 
The final step is just to draw everything on a single image. Here is how the final result looks like: 

| Final result   | 
|----------------|
|![][image27]    |


### Finding the car in videos
For the videos, the pipeline follows the basic pipeline applied to single images. Additionally, because of the temporal dimension, some additional filtering is applied. Here is what is done:
 1. The bounding boxes of all already detected cars are used when the heatmap is calculated. Those bounding boxes are regarded as if the car has been identified on that spot. That helps avoid flicker and loosing of already identified cars (lines 240-242)
 2. The bounding box is averaged over last 21 frames
 3. If the car is not found in 5 consecutive frames it is has disappeared. New cars, need to be found in 5 consecutive frames to be drawn and considered as existing (lies 77-105). 

The pipeline is run on both provided videos and works great. No false detections or not identifying existing cars occur. 

 1. [project_video](./output_videos/cars_project_video.mp4)
 2. [test_video](./output_videos/cars_test_video.mp4)


---

## Discussion

The described pipeline works great for the provided videos, but that needs to be thoroughly tested on more videos in changing lighting conditions. What I found interesting, is that there is a part in project video where two cars are classified as one. The first car partially occludes the second one, but it still gets classified as a car. The car didn't disappear, it is just occluded. More robust procedure regarding this issue has to be found. 

Calculating distance to the car works quite nice, even better than I have expected. Nevertheless, there are still some issues when the color of the road surface is changing, the removal of the road from the bottom of identified bounding box gives the false readings. Also, the speed is quite jumpy, so it too has to be filtered, but even in this form, it can give information of whether the detected car is closing or moving away from us.

The last thing is that this procedure is very time-consuming. On average it takes about 1.4 seconds per iteration (Ubuntu 16.06, 2xIntel(R) Core(TM) i7-4510U CPU @ 2.00GHz, 8GB DDR3) to detect cars and lanes. It is far from being real-time so that it can be employed in a real self-driving car. By profiling, I have noted that about 50% of the time goes to calculating histograms. This code needs to be optimized and maybe rewritten in C/C++.


