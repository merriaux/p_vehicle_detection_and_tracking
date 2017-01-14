import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time

from loadData import LoadData
from features import Features
from classif import Classifier
from slidingWindows import SlidingWindows
from sklearn.preprocessing import StandardScaler


class ProcessImage:
    def __init__(self):
        self.feat = Features()
        self.clf = Classifier()
        self.SlidWin = SlidingWindows()
        self.Windows = []

    def load(self,img):
        self.clf.load('svcModel01.pkl')
        self.Windows = self.SlidWin.pyramid_windows(img,(32,256))
        self.feat.loadScalerFromPickle("featuresSaver01.pkl")

    def testClassifier(self):
        feat = Features()
        feat.loadFromPickle("features01.pkl")
        t = time.time()
        print("accuracy on train", self.clf.eval(feat.X_train, feat.y_train))
        print("accuracy on test", self.clf.eval(feat.X_test, feat.y_test))
        t2 = time.time()

    def process(self,img):
        winCarDetected = []
        for win in self.Windows:
            imgExtracted = self.SlidWin.imgResize(self.SlidWin.windowsImgExtract(img, win))

            f = self.feat.extract_featuresImgNP(imgExtracted, cspace='HLS', spatial_size=(32, 32),
                                                hist_bins=32, hist_range=(0, 256))
            #plt.figure()
            #plt.imshow(imgExtracted)


            X = np.vstack((f)).astype(np.float64)

            # Apply the scaler to X
            scaled_X = self.feat.scalerTransform.transform(X)

            yPred=self.clf.predict(scaled_X);
            if(yPred==1):# and  self.clf.svc.decision_function(scaled_X)>350):
                winCarDetected.append(win)
                #print(self.clf.svc.decision_function(scaled_X))

            #print('featurelen = ', len(f), "X.shape", X.shape, 'y=', yPred)
        return winCarDetected



    def computeWeightImg(self,winCarDetected,weightWinCarDetected,shape):
        img = np.zeros(shape[0:2],dtype=np.float32)

        for i in range(0,len(winCarDetected)):
            window=winCarDetected[i]
            img[window[0][1]:window[1][1], window[0][0]:window[1][0], :] = img[window[0][1]:window[1][1],window[0][0]:window[1][0],:] + weightWinCarDetected[i];



        return img


    def process64x64img(self, img):

        f = self.feat.extract_featuresImgNP(img, cspace='HLS', spatial_size=(32, 32),
                                            hist_bins=32, hist_range=(0, 256))
        # plt.figure()
        # plt.imshow(imgExtracted)


        X = np.vstack((f)).astype(np.float64)


        # Apply the scaler to X
        scaled_X = self.feat.scalerTransform.transform(X)

        return self.clf.predict(scaled_X)


    def run(self):

        image = mpimg.imread('test_images/test1.jpg')
        self.load(image)
        print("nb windows total : ", len(self.Windows))
        print(self.feat.scalerTransform.mean_)
        print(self.feat.scalerTransform.scale_)

        self.testClassifier()

        files = glob.glob('test_images/test*.jpg')
        print(files)

        f, axarr = plt.subplots(int(len(files) / 2), 2)  # , sharex=True)
        i = 0
        j = 0
        for file in files:
            image = mpimg.imread(file)

            wins = self.process(image)

            print("nb windows found : ", len(wins))
            window_img = self.SlidWin.draw_boxes(image, wins, color=(0, 255, 0), thick=2)

            axarr[i, j].set_title(file)
            axarr[i, j].imshow(window_img)
            i = i + 1
            if (i == 3):
                i = 0
                j = j + 1
        plt.show()
    def run2(self):



        image = mpimg.imread('test_images/test1.jpg')
        self.load(image)
        print("nb windows total : ", len(self.Windows))

        self.testClassifier()

        files = glob.glob('test_imagesMano/*.png')


        for file in files:
            image = mpimg.imread(file)

            print(file,":",self.process64x64img(image))


    def run2(self):



        image = mpimg.imread('test_images/test1.jpg')
        self.load(image)
        print("nb windows total : ", len(self.Windows))

        self.testClassifier()

        files = glob.glob('test_imagesMano/*.png')


        for file in files:
            image = mpimg.imread(file)

            print(file,":",self.process64x64img(image))

    def run3(self):
        image = mpimg.imread('test_images/test1.jpg')
        self.load(image)

        data = LoadData()
        data.loadDataset()

        nbCarPred=0

        for file in data.notcars:
            # Read in each one by one
            image = mpimg.imread(file)
            nbCarPred = nbCarPred + self.process64x64img(image)

        print("noCar, wrong predictions",nbCarPred,"/",len(data.notcars))

        nbCarPred = 0

        for file in data.cars:
            # Read in each one by one
            image = mpimg.imread(file)
            nbCarPred = nbCarPred + self.process64x64img(image)

        print("Car, right predictions", nbCarPred, "/", len(data.cars))

if __name__ == "__main__":
    obj = ProcessImage()
    obj.run()