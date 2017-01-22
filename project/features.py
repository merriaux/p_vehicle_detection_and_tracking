import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
import pickle
import time
from loadData import LoadData
from skimage.io import imread


#From loadData class, compute features for each sample, the whole feature is normalize.
#The dataset is randomized and splited in train and test set.
#Dump the features result in pickle file.
#Also dump the scaler (sklearn.preprocessing.StandardScaler) in a pickle file.


class Features:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scalerTransform=None

    # Define a function to compute binned color features
    def bin_spatial(self,img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    # Define a function to compute color histogram features
    def color_hist(self,img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to return HOG features and visualization
    def get_hog_features(self,img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self,imgs, cspace='RGB', spatial_size=(32, 32),
                         hist_bins=32, hist_range=(0, 256), orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = imread(file)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #print("image ",image.dtype)
            features = features + self.extract_featuresImgNP(image, cspace, spatial_size,
                                  hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel)
        # Return list of feature vectors
        return features

    # extracture from a numpy array
    def extract_featuresImgNP(self,imgNP, cspace='RGB', spatial_size=(32, 32),
                         hist_bins=32, hist_range=(0, 256), orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):
        # Create a list to append feature vectors to
        features = []

        #imgNorm =
        img_lab = cv2.cvtColor(imgNP, cv2.COLOR_RGB2LAB)
        #print("imgNP ", imgNP.dtype,"img_lab ", img_lab.dtype )

        hog_array_L = hog(img_lab[:,:,0], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,visualise=False, feature_vector=True)
        hog_array_a = hog(img_lab[:,:,1], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,visualise=False, feature_vector=True)
        hog_array_b = hog(img_lab[:,:,2], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,visualise=False, feature_vector=True)
        hog_array = np.concatenate((hog_array_L, hog_array_a, hog_array_b), axis=0)
        #normalized = normalize(hog_array)
        features.append(hog_array)


        # Return list of feature vectors
        return features
    # compute feature for dataset car and no-car
    def featureCompute(self,cars_img,nocars_img):
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2

        car_features = self.extract_features(cars_img, cspace='HLS', spatial_size=(32, 32),
                                        hist_bins=32, hist_range=(0, 256))
        notcar_features = self.extract_features(nocars_img, cspace='HLS', spatial_size=(32, 32),
                                           hist_bins=32, hist_range=(0, 256))


        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        self.scalerTransform = StandardScaler().fit(X)


        # Apply the scaler to X
        scaled_X = self.scalerTransform.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        rand_state = np.random.randint(0, 100)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            scaled_X, y, test_size=0.15, random_state=rand_state)


    # feature size print
    def printFeaturesInfo(self):
        print("X_train shape", self.X_train.shape)
        print("y_train shape", self.y_train.shape)
        print("X_test shape", self.X_test.shape)
        print("y_test shape", self.y_test.shape)

    # save features to pickle file
    def saveToPickle(self,file):
        dict={"X_train":self.X_train,"y_train":self.y_train,"X_test":self.X_test,"y_test":self.y_test, "scaler":self.scalerTransform}
        with open(file, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #load features from pickle
    def loadFromPickle(self,file):
        with open(file, 'rb') as handle:
            d = pickle.load(handle)
            self.X_train = d["X_train"]
            self.y_train = d["y_train"]
            self.X_test = d["X_test"]
            self.y_test = d["y_test"]
            self.scalerTransform = d["scaler"]

    # save the scaler to pickle
    def saveScalerFromPickle(self, file):
        dict = {"scaler": self.scalerTransform}
        with open(file, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load scaler to pickle file
    def loadScalerFromPickle(self, file):
        with open(file, 'rb') as handle:
            d = pickle.load(handle)
            self.scalerTransform = d["scaler"]

    def run(self):
        t = time.time()
        data = LoadData()
        data.loadDataset()
        t2 = time.time()
        print(t2 - t, 'Seconds to load data.')

        t=time.time()
        self.featureCompute(data.cars,data.notcars)
        t2 = time.time()
        print(t2 - t, 'Seconds compute features.')
        self.printFeaturesInfo()
        t = time.time()
        self.saveToPickle("features01.pkl")
        self.saveScalerFromPickle("featuresSaver01.pkl")
        t2 = time.time()
        print(t2 - t, 'Seconds dump to file.')

    def runTestLoadScaler(self):
        self.loadScalerFromPickle("featuresSaver01.pkl")
        print(self.scalerTransform.mean_)
        print(self.scalerTransform.scale_)



if __name__ == "__main__":
    obj = Features()
    obj.run()
    obj.printFeaturesInfo()
    #obj.runTestLoadScaler();