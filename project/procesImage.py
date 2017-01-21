from object_tracker import Vehicle, Tracker

from moviepy.editor import VideoFileClip
import numpy as np
import pickle
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
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
from windowsReplay import WindowsReplay
import time

class ProcessImage:
    def __init__(self):
        self.feat = Features()
        self.clf = Classifier()
        self.SlidWin = SlidingWindows()
        self.Windows = []
        self.exportVideo = []
        self.windowsReplay = WindowsReplay()
        self.tracker = Tracker()
        self.frameNumber = 0

    def load(self,img):
        self.clf.load('svcModel01.pkl')
        self.Windows = self.SlidWin.pyramid_windows(img,(64,128))
        self.feat.loadScalerFromPickle("featuresSaver01.pkl")

    def debug_image2(self, img1, img2=None, img3=None, img4=None, img5=None, img6=None, img7=None):
        # Create an image to debug
        out_shape = (1080, 1920, 3)
        outImg = np.zeros(out_shape, dtype=np.uint8)
        if img1 is not None:
            if (len(img1.shape) == 3):
                outImg[0:720, 0:1280, :] = img1;
            else:
                outImg[0:720, 0:1280, 0] = img1 * 255;
                outImg[0:720, 0:1280, 1] = img1 * 255;
                outImg[0:720, 0:1280, 2] = img1 * 255;

        if img2 is not None:
            img2R = cv2.resize(img2, (640, 480), interpolation=cv2.INTER_CUBIC)
            if (len(img2R.shape) == 3):
                outImg[0:480, 1280:1280 + 640, :] = img2R;
            else:
                outImg[0:480, 1280:1280 + 640, 0] = img2R * 255;
                outImg[0:480, 1280:1280 + 640, 1] = img2R * 255;
                outImg[0:480, 1280:1280 + 640, 2] = img2R * 255;

        if img3 is not None:
            img3R = cv2.resize(img3, (640, 480), interpolation=cv2.INTER_CUBIC)
            if (len(img3R.shape) == 3):
                outImg[480:480 + 480, 1280:1280 + 640, :] = img3R;
            else:
                outImg[480:480 + 480, 1280:1280 + 640, 0] = img3R * 255;
                outImg[480:480 + 480, 1280:1280 + 640, 1] = img3R * 255;
                outImg[480:480 + 480, 1280:1280 + 640, 2] = img3R * 255;

        if img4 is not None:
            img4R = cv2.resize(img4, (320, 240), interpolation=cv2.INTER_CUBIC)
            if (len(img4R.shape) == 3):
                outImg[720:720 + 240, 0:320, :] = img4R;
            else:
                outImg[720:720 + 240, 0:320, 0] = img4R * 255;
                outImg[720:720 + 240, 0:320, 1] = img4R * 255;
                outImg[720:720 + 240, 0:320, 2] = img4R * 255;

        if img5 is not None:
            img5R = cv2.resize(img5, (320, 240), interpolation=cv2.INTER_CUBIC)
            if (len(img5R.shape) == 3):
                outImg[720:720 + 240, 0 + 320:320 + 320, :] = img5R;
            else:
                outImg[720:720 + 240, 0 + 320:320 + 320, 0] = img5R * 255;
                outImg[720:720 + 240, 0 + 320:320 + 320, 1] = img5R * 255;
                outImg[720:720 + 240, 0 + 320:320 + 320, 2] = img5R * 255;

        if img6 is not None:
            img6R = cv2.resize(img6, (320, 240), interpolation=cv2.INTER_CUBIC)
            if (len(img6R.shape) == 3):
                outImg[720:720 + 240, 0 + 320 + 320:320 + 320 + 320, :] = img6R;
            else:
                outImg[720:720 + 240, 0 + 320 + 320:320 + 320 + 320, 0] = img6R * 255;
                outImg[720:720 + 240, 0 + 320 + 320:320 + 320 + 320, 1] = img6R * 255;
                outImg[720:720 + 240, 0 + 320 + 320:320 + 320 + 320, 2] = img6R * 255;

        if img7 is not None:
            img7R = cv2.resize(img7, (320, 240), interpolation=cv2.INTER_CUBIC)
            if (len(img7R.shape) == 3):
                outImg[720:720 + 240, 0 + 320 + 320 + 320:320 + 320 + 320 + 320, :] = img7R;
            else:
                outImg[720:720 + 240, 0 + 320 + 320 + 320:320 + 320 + 320 + 320, 0] = img7R * 255;
                outImg[720:720 + 240, 0 + 320 + 320 + 320:320 + 320 + 320 + 320, 1] = img7R * 255;
                outImg[720:720 + 240, 0 + 320 + 320 + 320:320 + 320 + 320 + 320, 2] = img7R * 255;

        return (outImg)

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
        wins,b = self.FullFrameMultiScaleProcess(imgIn)
        window_img = self.SlidWin.draw_boxesNP(imgIn, wins, color=(0, 255, 0), thick=2)
        hot = self.hotImage(imgIn,wins,b)
        centers, boundingBox, result, binary_output = self.hot2boundingBox(hot,imgIn)
        imgHot = self.hot2uint8Img(hot)
        imgOut=self.debug_image2(result,window_img,imgHot,imgIn,binary_output)
        return imgOut

    def processImageVideoExportWindows(self,imgIn):
        wins,b = self.FullFrameMultiScaleProcess(imgIn)
        window_img = self.SlidWin.draw_boxesNP(imgIn, wins, color=(0, 255, 0), thick=2)
        hot = self.hotImage(imgIn,wins,b)
        centers, boundingBox, result, binary_output = self.hot2boundingBox(hot,imgIn)
        imgHot = self.hot2uint8Img(hot)
        imgOut=self.debug_image2(result,window_img,imgHot,imgIn,binary_output)
        dict={"windows":wins, "belief":b}
        self.exportVideo.append(dict)
        return imgOut

    def processImageVideoImportWindows(self,imgIn):
        wins=self.windowsReplay.exportVideo[self.frameNumber]["windows"]
        b = self.windowsReplay.exportVideo[self.frameNumber]["belief"]
        #print("frame : ",self.frameNumber)
        #print("wins : ", wins,"b",b)

        window_img = self.SlidWin.draw_boxesNP(imgIn, wins, color=(0, 255, 0), thick=2)
        hot = self.hotImage(imgIn,wins,b)
        centers, boundingBox, result, binary_output = self.hot2boundingBox(hot,imgIn)
        imgHot = self.hot2uint8Img(hot)
        imgOut=self.debug_image2(result,window_img,imgHot,imgIn,binary_output)
        self.frameNumber+=1
        return imgOut

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

    def processVideoExportWindows(self, fileIn, fileOut, start=0.0, end=0.0):
        print("Start Processing for exporting windows : ", fileIn)
        video_output = 'project_videoProcessed.mp4'
        if (start == 0.0 and end == 0.0):
            clip3 = VideoFileClip(fileIn)  # .subclip(40,42)
        else:
            clip3 = VideoFileClip(fileIn).subclip(start, end)
        yellow_clip1 = clip3.fl_image(self.processImageVideoExportWindows)
        yellow_clip1.write_videofile(fileOut, audio=False)

        print("Save to pickle")
        file = "windows01.pkl"
        with open(file, 'wb') as handle:
            pickle.dump(self.exportVideo, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print("End Processing, write : ", fileOut)

    def processVideoImportWindows(self, fileIn, fileOut, start=0.0, end=0.0):
        print("Start Processing with importing windows : ", fileIn)
        self.frameNumber = int(start*25); # quite approximative
        self.windowsReplay.load("windows01.pkl")
        video_output = 'project_videoProcessed.mp4'
        if (start == 0.0 and end == 0.0):
            clip3 = VideoFileClip(fileIn)  # .subclip(40,42)
        else:
            clip3 = VideoFileClip(fileIn).subclip(start, end)
        yellow_clip1 = clip3.fl_image(self.processImageVideoImportWindows)
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

    def runMultiScaleTest(self):

        image = imread('test_images/test1.jpg')
        self.load(image)


        self.testClassifier()

        files = glob.glob('test_images/test*.jpg')
        print(files)

        f, axarr = plt.subplots(int(len(files) / 2), 2)  # , sharex=True)
        i = 0
        j = 0
        for file in files:
            image = imread(file)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            t1 = time.time()
            wins,b = self.FullFrameMultiScaleProcess(image)
            t2 = time.time()
            print("time to compute ",file,":",t2-t1,"s")

            print("nb windows found : ", len(wins))
            window_img = self.SlidWin.draw_boxesNP(image, wins, color=(0, 255, 0), thick=2)

            t1 = time.time()
            hot = self.hotImage(image, wins, b)
            t2 = time.time()
            print("hot image generation time : ", t2 - t1, "s")

            binary_output = np.zeros_like(hot)
            binary_output[hot > 3.0] = 1
            binary_zeros = np.zeros_like(binary_output, dtype=np.uint8)
            color_warp = np.dstack((binary_zeros, binary_output.astype(np.uint8) * 255, binary_zeros))
            result = cv2.addWeighted(image, 1, color_warp, 0.3, 0)

            imgBox = self.SlidWin.draw_boxesNP(image, wins, color=(255, 0, 0), thick=2)
            if(np.sum(binary_output)>1):
                ret, thresh = cv2.threshold(binary_output.astype(np.uint8) * 255, 127, 255, 0)
                im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(result, (cx, cy), 10, (0, 0, 255), 5)


            axarr[i, j].set_title(file)
            axarr[i, j].imshow(result)
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

    def runVideo(self,outputFile,start=0.0,end=0.0):

        image = imread('test_images/test1.jpg')
        self.load(image)
        print("nb windows total : ", len(self.Windows))

        self.processVideoImportWindows("project_video.mp4",outputFile,start,end)
        #self.processVideo("project_video.mp4", outputFile, start, end)


    def FullFrameMultiScaleProcess(self, rawImg):
        winAll=[];
        beliefAll=[]
        img_lab = cv2.cvtColor(rawImg, cv2.COLOR_RGB2LAB)
        scales=[1.0,1.3,1.7,2.2,2.9]
        for scale in scales:
            topClip = int(350/scale)
            imgResized = cv2.resize(img_lab,(int(img_lab.shape[1]/scale),int(img_lab.shape[0]/scale)),interpolation = cv2.INTER_CUBIC)
            #print("imgResized.shape", imgResized.shape)
            f, windows = self.fullFrameHogAnalyse(imgResized, topClip, scale, 1)
            if(len(f)>0):
                #print("f:",f)
                #print("f.len", len(f))
                X = np.vstack(f).astype(np.float64)
                #print("X.shape", X.shape)
                #print("windows.shape", windows.shape)
                scaled_X = self.feat.scalerTransform.transform(X)
                yPred = self.clf.predict(scaled_X)
                yBelief = self.clf.svc.decision_function(scaled_X)
                winKeeped = windows[np.logical_and(yPred==1 , yBelief>0.6)]
                beliefAll.append(yBelief[np.logical_and(yPred==1 , yBelief>0.6)])
                #print(yBelief[np.logical_and(yPred==1 , yBelief>0.6)])
                #print("winKeeped.shape", winKeeped.shape)
                winAll.append(winKeeped)
                #print(winKeeped)
                #imgBox = self.SlidWin.draw_boxesNP(rawImg, windows, color=(0, 255, 0), thick=1)
                #plt.figure()
                #plt.imshow(imgBox)
                #imgBox = self.SlidWin.draw_boxesNP(rawImg, winKeeped, color=(0, 255, 0), thick=2)
                #plt.figure()
                #plt.imshow(imgBox)

        winKeepedNP = np.vstack(winAll).astype(np.int16)
        beliefKeepedNP=np.hstack(beliefAll)
        #print("winKeepedNP",winKeepedNP)
        #print("winKeepedNP.shape",winKeepedNP.shape)
        return winKeepedNP,beliefKeepedNP

    def hotImage(self,img, windows,belief):
        hot = np.zeros(img.shape[:2],dtype=np.float32)
        for i in range(0,windows.shape[0]):
            win = windows[i,:]
            hot[win[1]:win[3],win[0]:win[2]] += belief[i]
        return(hot)
    def hot2boundingBox(self,hot,image, hotThresh=3.0):
        binary_output = np.zeros_like(hot)
        binary_output[hot > hotThresh] = 1
        binary_zeros = np.zeros_like(binary_output, dtype=np.uint8)
        color_warp = np.dstack((binary_zeros, binary_output.astype(np.uint8) * 255, binary_zeros))
        result = cv2.addWeighted(image, 1, color_warp, 0.3, 0)
        ret, thresh = cv2.threshold(binary_output.astype(np.uint8) * 255, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers=[]
        boundingBox=[]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            M = cv2.moments(cnt)
            if(M['m00']!=0):
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append([cx,cy])
                boundingBox.append([x, y, w, h])
                cv2.circle(result, (cx, cy), 10, (0, 0, 255), 5)
        return centers, boundingBox, result, binary_output
    def hot2uint8Img(self,hot):

        imgHot = cm.hot(hot / 10)
        imgHot = (imgHot[:, :, :3] * 255).astype(np.uint8)
        return imgHot

    def testFullFrameProcess2(self):


        image = imread('test_images/test1.jpg')
        self.load(image)
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        t1=time.time()
        windows,b=self.FullFrameMultiScaleProcess(image)
        t2 = time.time()
        print("process time : ", t2-t1,"s")

        t1 = time.time()
        hot = self.hotImage(image,windows,b)
        t2 = time.time()
        print("hot image generation time : ", t2 - t1, "s")

        binary_output = np.zeros_like(hot)
        binary_output[hot>3.0]=1
        binary_zeros = np.zeros_like(binary_output,dtype=np.uint8)
        color_warp = np.dstack((binary_zeros, binary_output.astype(np.uint8)*255, binary_zeros))
        result = cv2.addWeighted(image, 1, color_warp, 0.3, 0)
        imgBox = self.SlidWin.draw_boxesNP(image, windows, color=(255, 0, 0), thick=2)
        ret, thresh = cv2.threshold(binary_output.astype(np.uint8)*255, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(result, (cx,cy), 10, (0, 0, 255), 5)
            #print("position : ",cx,cy)

        import matplotlib as mpl
        import matplotlib.cm as cm
        imgHot = cm.hot(hot/10)
        print(imgHot.shape)
        imgHot = (imgHot[:,:,:3]*255).astype(np.uint8)

        plt.figure()
        plt.imshow(imgBox)
        plt.figure()
        plt.imshow(hot)
        plt.figure()
        plt.imshow(binary_output)
        plt.figure()
        plt.imshow(result)
        plt.figure()
        plt.imshow(imgHot)
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
    obj.runVideo("video01.mp4",34,35)
    #obj.testFullFrameProcess2()
    #obj.runMultiScaleTest()