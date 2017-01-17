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
            image = mpimg.imread(file)
            features = features + self.extract_featuresImgNP(image, cspace, spatial_size,
                                  hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel)
        # Return list of feature vectors
        return features
    def extract_featuresImgNP(self,imgNP, cspace='RGB', spatial_size=(32, 32),
                         hist_bins=32, hist_range=(0, 256), orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):
        # Create a list to append feature vectors to
        features = []

        # Read in each one by one
        image = imgNP
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else:
            feature_image = np.copy(image)
        gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        # Apply bin_spatial() to get spatial color features
        '''
        spatial_features = self.bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = self.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Call get_hog_features() with vis=False, feature_vec=True
        #t1 = time.time()
        '''
        hog_features = self.get_hog_features(gray_image, orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        #t2 = time.time()
        #print(t2-t1,"s")
        '''
        otherHogImage = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        hog_featuresH = self.get_hog_features(otherHogImage[:,:,0], orient,
                                             pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        hog_featuresL = self.get_hog_features(otherHogImage[:, :, 1], orient,
                                              pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        hog_featuresS = self.get_hog_features(otherHogImage[:, :, 2], orient,
                                              pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        '''
        # Append the new feature vector to the features list

        hog_array = np.concatenate((hog_features,), axis = 0)
        #normalized = normalize(hog_array.reshape(1, -1))
        features.append(hog_array)
        #features.append((hog_features,hog_featuresL,hog_featuresA,hog_featuresB), axis = 0)
        #print("feature shape",hog_features.shape, hog_featuresL.shape, " hog ", hog_featuresA.shape,hog_featuresB.shape)


        # Return list of feature vectors
        return features

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



    def printFeaturesInfo(self):
        print("X_train shape", self.X_train.shape)
        print("y_train shape", self.y_train.shape)
        print("X_test shape", self.X_test.shape)
        print("y_test shape", self.y_test.shape)


    def saveToPickle(self,file):
        dict={"X_train":self.X_train,"y_train":self.y_train,"X_test":self.X_test,"y_test":self.y_test, "scaler":self.scalerTransform}
        with open(file, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def loadFromPickle(self,file):
        with open(file, 'rb') as handle:
            d = pickle.load(handle)
            self.X_train = d["X_train"]
            self.y_train = d["y_train"]
            self.X_test = d["X_test"]
            self.y_test = d["y_test"]
            self.scalerTransform = d["scaler"]


    def saveScalerFromPickle(self, file):
        dict = {"scaler": self.scalerTransform}
        with open(file, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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