
# Project sources files description

The project is composed of different python class :

- **loadData.py** : 
Able to explorer folder of the whole dataset, to set list of vehicles and no-vehicles samples.
There are few functions to explore the dataset as display images, statistics about class...

- **features.py** :
From *loadData*, compute features for each sample, the whole feature is normalize. 
The dataset is randomized and splited in train and test set.
Dump the features result in pickle file. 
Also dump the scaler (sklearn.preprocessing.StandardScaler) in a pickle file.

- **classif.py** :
From *features*, implement SVC to classify features. 
The SVC class is saved in file with sklearn.externals.joblib
few functions, to compute the score on test and train set, display confusion matrix and plot SVC coefficients and feature index. (To determine each ones are significant).

- **slidingWindows.py** : 
Sliding windows implementation, with region of interest in the image. 
Pyramid windows, with size step and overlap tuning parameter. 
Extraction and resizing of all image windows for prediction in classif class. 
We will see later that because of compute effiency this implementation wasn't used in the video processing.

- **procesImage.py** :
All the processing step for test images, and video conversion.


- **windowsReplay.py** :
To experimente the tracking algorithm, I have recorded all the sliding windows predited car for each video frame in pickle file, and use it to tume tracking parameter.

- ** object_tracker.py** :
Vehicles tracking implementation with a GNN algorithm (Global nearest neighbor  http://www.control.isy.liu.se/student/graduate/TargetTracking/Lecture5.pdf).

- **laneDetection.py** :
Few class to reuse project 4 lane detection.


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
I use linearSCV classifier. The feature are normalize with *sklearn.preprocessing.StandardScaler*. I have tried SCV(rbf), the score was better but the time to fit and predit was really huge. 
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


## Windows merging
All the windows for the same car have to be merge in only one bounding box.

For that I use a "*hot point*" image. In a float image, I add the *decision_function* on each windows. And I obtain this kind of result :
![Origine image](readmeImg/ProcessTest1.png)
![Hot point](readmeImg/hotpoint.png)

I threshold it:
![Hot point thresholded](readmeImg/hotpointThreshold.png)

I use the *cv2.findcontours* function to extract shape contours. *cv2.boundingRect* return me the bouding box, 
and cv2.moments able me to compute the centroides.

The results on test images is below (green : adding of thresholded hot point):

![Final results on test images](readmeImg/multiScaleResultHotpoint.png)

## video processing

I have processed the project video with this pipeline and the result is not soo bad. The processed video is here: https://www.youtube.com/watch?v=PLFG7eKJ17o

It have some time a false positive, so I will implement a tracking solution for label the car and remove if their age are young.

## tracking
I implement a GNN algorithm (Global nearest neighbor  http://www.control.isy.liu.se/student/graduate/TargetTracking/Lecture5.pdf).
For all targets predicted in hotpoint image, I test the shortest distance with actual cars tracked position. (test all the combination : not very efficience if there are lots of cars in the image).
If this distance is below a threshold, I increase the age of the car. 
If the car not have target the age is decrease.
If targets don't find associated car, I create a new car object to track this new target.

If the age is above a threshold the car is display, that filter short time false positive detections. If the target isn't detected uring few frames, this age decreace since 0, so the car isn't delete immediatly. So the target will detect few frames after, the car will have the same trakcing index.
Now we are able to have a "unique index number" fr each vehile tracked. 
So we can average the position and the bounding box of each car on time. 

## lane detection
I just refactorize my project 4 code in *LaneDetection* class, and run it from input image.

# final video
The final result seem to be nice. 
The video file is pushed in Git : *videoOutputFinalProject.mp4*
It could be also find here : https://www.youtube.com/watch?v=GRynrQc-2_M


Decription of each "debug video" patchwork :
1. The final output with averaged tracking position/bounding box and car labeling
2. The ouput of fincontour and centroide extraction (from hotPoint segmentation)
3. The predicted sliding windows result
4. Hotpoint image
5. Segment hotpoint image
6. The original input image
7. Lines search (project 4)
8. Unwrapped perspective image (project 4)
9. Global histogram research (project 4)
10. Image line extraction (project 4)



![Result video description](readmeImg/finalVideoDescription.png)

Tracking could be improve with quite logic, to don't merge two vehicles in same one when they are Superimposed. 


