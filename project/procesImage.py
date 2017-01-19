from moviepy.editor import VideoFileClip
import numpy as np
import cv2

from skimage.feature import hog

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time
from skimage.io import imread
from loadData import LoadData
from features import Features
from classif import Classifier
from slidingWindows import SlidingWindows
from sklearn.preprocessing import StandardScaler
import time

class ProcessImage:
    def __init__(self):
        self.feat = Features()
        self.clf = Classifier()
        self.SlidWin = SlidingWindows()
        self.Windows = []

    def load(self,img):
        self.clf.load('svcModel01.pkl')
        self.Windows = self.SlidWin.pyramid_windows(img,(64,128))
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
            #print("img", img.dtype)
            #print("imgExtracted", imgExtracted.dtype)
            f = self.feat.extract_featuresImgNP(imgExtracted, cspace='HLS', spatial_size=(32, 32),
                                                hist_bins=32, hist_range=(0, 256))
            #plt.figure()
            #plt.imshow(imgExtracted)


            X = np.vstack((f)).astype(np.float64)

            # Apply the scaler to X
            scaled_X = self.feat.scalerTransform.transform(X)
            #print("scaled_X.shape=",scaled_X.shape)
            yPred=self.clf.predict(scaled_X);
            if((yPred==1) and self.clf.svc.decision_function(scaled_X)>0.1):#0.2):
                winCarDetected.append(win)
                #print(self.clf.svc.decision_function(scaled_X))
                #print(self.clf.svc.decision_function(scaled_X))

            #print('featurelen = ', len(f), "X.shape", X.shape, 'y=', yPred)
        return winCarDetected



    def computeWeightImg(self,winCarDetected,weightWinCarDetected,shape):
        img = np.zeros(shape[0:2],dtype=np.float32)

        for i in range(0,len(winCarDetected)):
            window=winCarDetected[i]
            img[window[0][1]:window[1][1], window[0][0]:window[1][0], :] = img[window[0][1]:window[1][1],window[0][0]:window[1][0],:] + weightWinCarDetected[i];



        return img

    # image already converted to LAB color space
    def fullFrameHogAnalyse(self,img,topClip=0,scale=1.0,step=1):
        features =[]
        windows=[]
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        # just a first one, to know the size
        hog_array_L = hog(img[topClip:, :, 0], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                          cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=False,
                          feature_vector=False)
        nbWindowsMax = len(range(0, hog_array_L.shape[0]-7,step))*len(range(0, hog_array_L.shape[1]-7,step))

        win = np.zeros((nbWindowsMax,4),dtype=np.int16)


        hog_array_L = hog(img[topClip:, :, 0], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                          cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=False,
                          feature_vector=False)
        hog_array_A = hog(img[topClip:, :, 1], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                          cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=False,
                          feature_vector=False)
        hog_array_B = hog(img[topClip:, :, 2], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                          cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=False,
                          feature_vector=False)
        winIdx=0
        for i in range(0, hog_array_L.shape[0]-7,step):
            for j in range(0, hog_array_L.shape[1]-7,step):
                win[winIdx, 0] = np.int16((j * 8)*scale)
                win[winIdx, 1] = np.int16((topClip+i * 8)*scale)
                win[winIdx, 2] = np.int16((j * 8 + 64)*scale)
                win[winIdx, 3] = np.int16((topClip+i * 8 + 64)*scale)
                #TODO problem de positionnment des fenetres avec le multi-echelle --> problem ci dessous
                if(self.SlidWin.isWindowsInRoiNP(win[winIdx,:])):
                    fL = hog_array_L[i:i + 7, j:j + 7, :, :, :].ravel();
                    fA = hog_array_A[i:i + 7, j:j + 7, :, :, :].ravel();
                    fB = hog_array_B[i:i + 7, j:j + 7, :, :, :].ravel();
                    hog_array = np.concatenate((fL, fA, fB), axis=0)
                    features.append(hog_array)
                    winIdx = winIdx + 1

        win = win[0:winIdx,:]
        return features, win

    def process64x64img(self, img):

        f = self.feat.extract_featuresImgNP(img, cspace='HLS', spatial_size=(32, 32),
                                            hist_bins=32, hist_range=(0, 256))
        # plt.figure()
        # plt.imshow(imgExtracted)


        X = np.vstack((f)).astype(np.float64)


        # Apply the scaler to X
        scaled_X = self.feat.scalerTransform.transform(X)

        return self.clf.predict(scaled_X)

    def processImageVideo(self,imgIn):
        wins = self.process(imgIn)
        window_img = self.SlidWin.draw_boxes(imgIn, wins, color=(0, 255, 0), thick=2)
        return window_img


    def processVideo(self,fileIn, fileOut,start=0.0,end=0.0):
        print("Start Processing : ", fileIn)
        video_output = 'project_videoProcessed.mp4'
        if(start==0.0 and end==0.0):
            clip3 = VideoFileClip(fileIn)#.subclip(40,42)
        else:
            clip3 = VideoFileClip(fileIn).subclip(start,end)
        yellow_clip1 = clip3.fl_image(self.processImageVideo)
        yellow_clip1.write_videofile(fileOut, audio=False)
        print("End Processing, write : ", fileOut)


    def run(self):

        image = imread('test_images/test1.jpg')

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        self.load(image)
        print("nb windows total : ", len(self.Windows))
        print(self.feat.scalerTransform.mean_)
        print(self.feat.scalerTransform.scale_)

        plt.figure()
        window_img = self.SlidWin.draw_boxes(image, self.Windows, color=(0, 255, 0), thick=2)
        plt.imshow(window_img)

        self.testClassifier()

        files = glob.glob('test_images/test*.jpg')
        print(files)

        f, axarr = plt.subplots(int(len(files) / 2), 2)  # , sharex=True)
        i = 0
        j = 0
        for file in files:
            image = imread(file)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("image origine", image.dtype)
            t1 = time.time()
            wins = self.process(image)
            t2 = time.time()
            print("time to compute ",file,":",t2-t1,"s")

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
        image = imread('test_images/test1.jpg')
        self.load(image)
        print("nb windows total : ", len(self.Windows))

        self.testClassifier()

        files = glob.glob('test_imagesMano/*.png')


        for file in files:
            image = imread(file)

            print(file,":",self.process64x64img(image))

    def run3(self):
        image = imread('test_images/test1.jpg')
        self.load(image)

        data = LoadData()
        data.loadDataset()

        nbCarPred=0

        for file in data.notcars:
            # Read in each one by one
            image = imread(file)
            nbCarPred = nbCarPred + self.process64x64img(image)

        print("noCar, wrong predictions",nbCarPred,"/",len(data.notcars))

        nbCarPred = 0

        for file in data.cars:
            # Read in each one by one
            image = imread(file)
            nbCarPred = nbCarPred + self.process64x64img(image)

        print("Car, right predictions", nbCarPred, "/", len(data.cars))

    def runVideo(self,outputFile,start,end):

        image = imread('test_images/test1.jpg')
        self.load(image)
        print("nb windows total : ", len(self.Windows))

        self.processVideo("project_video.mp4",outputFile,start,end)

    def testFullFrameMultiScaleProcess(self, rawImg):
        winAll=[];
        img_lab = cv2.cvtColor(rawImg, cv2.COLOR_RGB2LAB)
        scales=[1.0,1.3,1.7,2.2,2.9]
        for scale in scales:
            topClip = int(350/scale)
            imgResized = cv2.resize(img_lab,(int(img_lab.shape[1]/scale),int(img_lab.shape[0]/scale)),interpolation = cv2.INTER_CUBIC)
            print("imgResized.shape", imgResized.shape)
            f, windows = self.fullFrameHogAnalyse(imgResized, topClip, scale, 1)
            if(len(f)>0):
                #print("f:",f)
                print("f.len", len(f))
                X = np.vstack(f).astype(np.float64)
                print("X.shape", X.shape)
                print("windows.shape", windows.shape)
                scaled_X = self.feat.scalerTransform.transform(X)
                yPred = self.clf.predict(scaled_X)
                yBelief = self.clf.svc.decision_function(scaled_X)
                winKeeped = windows[np.logical_and(yPred==1 , yBelief>0.1)]
                print("winKeeped.shape", winKeeped.shape)
                winAll.append(winKeeped)
                print(winKeeped)
                imgBox = self.SlidWin.draw_boxesNP(rawImg, windows, color=(0, 255, 0), thick=1)
                plt.figure()
                plt.imshow(imgBox)
                imgBox = self.SlidWin.draw_boxesNP(rawImg, winKeeped, color=(0, 255, 0), thick=2)
                plt.figure()
                plt.imshow(imgBox)

        winKeepedNP = np.vstack(winAll).astype(np.int16)
        print("winKeepedNP",winKeepedNP)
        print("winKeepedNP.shape",winKeepedNP.shape)
        return winKeepedNP

    def testFullFrameProcess2(self):


        image = imread('test_images/test1.jpg')
        self.load(image)
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        t1=time.time()
        windows=self.testFullFrameMultiScaleProcess(image)
        t2 = time.time()
        print("process time : ", t2-t1,"s")
        imgBox = self.SlidWin.draw_boxesNP(image, windows, color=(255, 0, 0), thick=2)
        plt.figure()
        plt.imshow(imgBox)

        plt.show()



    def testFullFrameProcess(self):
        print(np.int16(12.5))

        image = imread('test_images/test1.jpg')
        self.load(image)
        self.testClassifier()

        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        t1=time.time()
        topClip = 350
        scale = 1.0
        f,windows=self.fullFrameHogAnalyse(img_lab,topClip,scale,2)
        t2 = time.time()
        print("feature extraction",t2-t1,"s")
        t1 = time.time()
        X = np.vstack(f).astype(np.float64)
        print("X.shape",X.shape)
        print("windows.shape", windows.shape)
        # Apply the scaler to X
        scaled_X = self.feat.scalerTransform.transform(X)
        # print("scaled_X.shape=",scaled_X.shape)
        yPred = self.clf.predict(scaled_X)
        yBelief = self.clf.svc.decision_function(scaled_X)
        t2 = time.time()
        print("prediction", t2 - t1, "s")
        print("yPred.shape", yPred.shape)
        print("yBelief.shape", yBelief.shape)
        print("nb win",np.sum(yPred))
        print(yBelief[yPred==1])
        print(windows)
        imgBox = self.SlidWin.draw_boxesNP(image, windows, color=(0, 255, 0), thick=1)
        plt.figure()
        plt.imshow(imgBox)

        imgBox = self.SlidWin.draw_boxesNP(image, windows[np.logical_and(yPred==1 , yBelief>0.1)], color=(0, 255, 0), thick=2)
        plt.figure()
        plt.imshow(imgBox)
        plt.show()
if __name__ == "__main__":
    obj = ProcessImage()
    #obj.run()
    #obj.runVideo("video01.mp4",23,24)
    obj.testFullFrameProcess2()