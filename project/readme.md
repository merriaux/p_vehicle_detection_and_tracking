
# Project sources files 

The project is composed of different python class :

- **loadData.py** : 
Able to explorer folder of the whole dataset, to set list of vehicles and no-vehicles samples.
There are few functions to explore the dataset as display images, statistics about class...

- **features.py** :
From *loadData*, compute features for each sample, the whole feature is normalize. 
The dataset is randomized and splited in train and test set.
Store the features result in pickle file. 
Also store the normalizer in a pickle file.

- **classif.py** :
From *features*, implement SVC to classify features. 
The SVC class is saved in file with sklearn.externals.joblib
few functions, to compute the score on test and train set, display confusion matrix and plot SVC coefficients and feature index. (To détermine each ones are significant).

- **slidingWindows.py** : 
Sliding windows implementation, with region of interest in the image. 
Pyramid windows, with size step and overlap tuning parameter. 
Extraction and resizing of all image windows for prediction in classif class. 
We will see later that because of compute effiency this implementation wasn't used in the video processing.

- **procesImage.py** :
All the processing step for test images, and video conversion.


- **windowsReplay.py** :
To experimente the tracking algorithm, I have recorded all the sliding windows predited car for each video frame in pickle file, and use it to tume tracking parameter.

# Information to run the project

If you want the run the project, run in this order :

1. features.py
2. classif.py
3. procesImage.py


# Project conception and details

## features extraction	
I have tested different	kinds of features (color histogramme, reduce size image, histogramme of gradient), in few color spaces (RGB, HSV, LUV, HLS, YUV and LAB). 
I have obtain my best result with LAB color space and a HOG on each channel. 
The feature size is 5292 for a image of 64x64 pixels. 

My parameters are : orient=9, pix_per_cell=8, cell_per_block=2

## Classification
I use linearSCV classifier. I have try SCV(rbf), the score was better but the time to fit and predit was really huge. 
So the linearSCV seem to be a good compromise.

With C=1 the result is :
- accuracy on train 1.0
- accuracy on test 0.978

I test different value for C parameter to increase score on test C, and increase its generalization capacity.

My best result is with C=0.0001 :

- Training duration 2.66s
- accuracy on train 0.997
- accuracy on test 0.992
- confusion matrix train set: 

 [[8192   14]
 
 [  30 7455]]
- confusion matrix test set: 

 [[1450   13]
 
 [   9 1298]]


## Sliding windows
In a first, I generate sliding pyramid windows from size 32x32, 64x64, 96x96 and 128x128. 
The windows fully outside region of interest are not append to reduce their number.
The overlap is 0.5. I obtain 2214 windows to process.

Example of the 2214 sliding windows with ROI :

![The classic pyramid windows solution](readmeImg/classicPyramidWindowsParamStd.png)

All the windows are resize to 64x64 pixels and Features extraction (LAB conversion and HOG) in the same matrix. 
Then I normalize it and predict with SVC. All this operations are really slow, around 10s per image. 
So I try a other idea.

# sliding windows optimized

the idea is the compute the HOG on the whole image and slide windows in feature directly (without ravel !).
For a 64x64 image, a single channel HOG feature shape is [7,7,2,2,9] x 3 channels.
So in the whole HOG image a select [7,7,2,2,9] features and reshape its to line (with *.ravel()* function).
I sliding this "windows" for each pix_per_cell, so the overlap is 7/8. I concatenate all in one matrix, to normalize and predict.
For multiscale search, I resize the whole image before HOG extraction.
Actualy I do it for 5 different scale factors : [1.0,1.3,1.7,2.2,2.9] it correspond to 64x64, 83x83, 108x108, 140x140 and 185x185 window sizes.

Of course I kept the ROI to reduce the number of windows.
With this parameters, the total number of windows is 10898 (A huge overlap !).
The cpu time to compute one image (1280x720) is around 2s, so really more efficency than my first sliding windows/features implemantation.


To reduce the number of false positif, I compute and threshold on *decision_function*.

The result on test images is below:
![The detection result on test images is below](readmeImg/multiScaleResult.png)







Criteria 	Meets Specifications

Have the HOG features been extracted from the training images?
	

The skimage.features.hog() function or other method has been used to extract HOG features from the labeled training images provided.

Were the parameters used to extract HOG features explained / justified?
	

The reasoning for the choices of parameters used for HOG feature extraction (orientations, pixels_per_cell, cells_per_block) has been explained.

Has a classifier been trained using HOG features (and optionally additional color/histogram features)?	
	
The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier.

Sliding Window Search
Criteria 	Meets Specifications

Has a sliding-window technique been implemented to search for vehicles in the test images?
	

A sliding window approach has been implemented, where overlapping tiles in each test image are classified as vehicle or non-vehicle. Some justification has been given for the particular implementation chosen.

Video Implementation
Criteria 	Meets Specifications

Has the pipeline been used to process the example videos and identify vehicles in each frame?
	

The sliding-window search plus classifier has been used to search for and identify vehicles in the videos provided. Video output has been generated with detected vehicle positions drawn on each frame of video.

Has some sort of filtering mechanism been implemented to reject false positives? Does the method reduce the number of false positives?
	

A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives.

README
Criteria 	Meets Specifications

Has a Readme file been included that describes in detail the steps taken to construct the pipeline, techniques used, areas where improvements could be made?
	

The Readme file submitted with this project includes a detailed description of what steps were taken to achieve the result, what techniques were used to arrive at a successful result, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.


Run :
- features.py
- classif.py
- processImage.py