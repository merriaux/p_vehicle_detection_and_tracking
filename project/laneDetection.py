import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

from skimage.io import imread

class Line2():
    def __init__(self, numberFrame=5):
        self.poly = []
        self.n = numberFrame
        self.lastPoly = None
        self.lastLineOk = False
        self.curvature = []
        self.position = []

    def notDetected(self):
        self.lastLineOk = False

    def addPoly(self, newPoly):
        self.lastLineOk = True
        self.lastPoly = newPoly
        # print(self.lastPoly)
        self.poly.append(newPoly)
        if (len(self.poly) > self.n):
            self.poly.pop(0)

    def polyMean(self):
        return (np.mean(np.array(self.poly), axis=0))

    def computeValue(self, yValues):
        p = self.polyMean()
        return (p[0] * yValues ** 2 + p[1] * yValues + p[2])

    def computeValueWithLastPoly(self, yValues):
        p = self.lastPoly
        # print("y :",yValues)
        # print("poly : ",p)
        return (p[0] * yValues ** 2 + p[1] * yValues + p[2])

    def addCurvatureAndPosition(self, cur, pos):
        self.curvature.append(cur)
        if (len(self.curvature) > self.n):
            self.curvature.pop(0)

        self.position.append(pos)
        if (len(self.position) > self.n):
            self.position.pop(0)

    def computeCurvatureAndPosition(self):
        return (np.mean(np.array(self.curvature)), np.mean(np.array(self.position)))


class Param():
    def __init__(self):
        self.lineL = Line2(10)  # 25)
        self.lineR = Line2(10)  # 25)

    def addCurvatureAndPosition(self, curL, posL, curR, posR):
        self.lineL.addCurvatureAndPosition(curL, posL)
        self.lineR.addCurvatureAndPosition(curR, posR)

    def computeCurvatureAndPosition(self):
        curL, posL = self.lineL.computeCurvatureAndPosition()
        curR, posR = self.lineR.computeCurvatureAndPosition()
        return ((curL + curR) / 2.0, (posL + posR) / 2)


class LaneDetection:
    def __init__(self):
        images = glob.glob('camera_cal/calibration*.jpg')
        img = cv2.imread(images[0])
        self.imshape = img.shape
        self.vertices = np.array([[(0 + 200, self.imshape[0] - 40), (138, 570), (560, 440),
                              (self.imshape[1] - 560, 440), (1131, 570), (self.imshape[1] - 200, self.imshape[0] - 40)]],
                            dtype=np.int32)
        self.laneWidth = 600
        self.laneHeight = img.shape[0] * 0.15

        self.xMaxLast = None

        self.params = Param()

    def cameraCalibration(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                print('OK :', fname)
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                write_name = 'camera_cal/corners_found' + str(idx) + '.jpg'
                cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
                cv2.waitKey(100)
            else:
                print('KO :', fname)

        cv2.destroyAllWindows()




        img_size = (img.shape[1], img.shape[0])
        # Do camera calibration given object points and image points
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump(dist_pickle, open("camera_cal/calibrationCam.p", "wb"))

        print("Images shape :\n", img_size)
        print("matrix :\n", self.mtx)
        print("dist ortions\n", self.dist)


    def loadCalibration(self):
        dataPkl = pickle.load(open("camera_cal/calibrationCam.p", "rb"))
        self.mtx = dataPkl["mtx"]
        self.dist = dataPkl["dist"]

        print("matrix :\n", self.mtx)
        print("distortions\n", self.dist)

    def calibrationCorrection(self,img):
        out = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return (out)

    def region_of_interest(self,img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def clipROI(self,img):
        return self.region_of_interest(img, self.vertices)

    def hls_select(self,img, thresh=(0, 255)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s = img[:, :, 2]
        # retval, binary_output = cv2.threshold(s.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        binary_output = np.zeros_like(s)
        binary_output[(s >= thresh[0]) & (s <= thresh[1])] = 1
        return binary_output, s

    def hls_selectH(self,img, thresh=(0, 255)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h = img[:, :, 0]
        # retval, binary_output = cv2.threshold(s.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        binary_output = np.zeros_like(h)
        binary_output[(h >= thresh[0]) & (h <= thresh[1])] = 1
        return binary_output, h

    def abs_sobel_thresh(self,img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        if (orient == 'x'):
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobelx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        # print("pierre")
        # print("scaled_sobel.min",np.min(scaled_sobel))
        # print("scaled_sobel.max",np.max(scaled_sobel))
        # retval, grad_binary = cv2.threshold(scaled_sobel.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output, scaled_sobel

    def mag_thresh(self,img, sobel_kernel=3, mag_thresh=(0, 255)):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return binary_output, gradmag

    def dir_threshold(self,img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # print(sobelx,sobely)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        # Here I'm suppressing annoying error messages
        binary_output = np.zeros_like(sobelx.astype(np.uint8))
        with np.errstate(divide='ignore', invalid='ignore'):
            # absgraddir = np.absolute(np.arctan(sobely/sobelx))
            absgraddir = np.absolute(np.arctan(np.absolute(sobely) / np.absolute(sobelx)))
            # print(absgraddir)
            scale_factor = np.max(absgraddir[~np.isnan(absgraddir)]) / 255
            # print(scale_factor)

            binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
            absgraddir = (absgraddir / scale_factor).astype(np.uint8)
        return binary_output, absgraddir

    def sobelExtract(self,img, thresh=(0, 255)):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sxbinary

    def linesExtraction(self,imgCorrected):
        # imgSbin,s = hls_select(imgCorrected,(150,255))
        imgSbin, s = self.hls_select(imgCorrected, (90, 255))
        imgSbinH, h = self.hls_selectH(imgCorrected, (15, 100))
        ksize = 5  # Choose a larger odd number to smooth gradient measurements

        gray = cv2.cvtColor(imgCorrected, cv2.COLOR_RGB2GRAY)

        gradx_binary, gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=5, thresh=(40, 100))  # thresh=(15, 50))
        grady_binary, grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=5, thresh=(20, 50))

        mag_binary, mag = self.mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 200))
        dir_binary, direction = self.dir_threshold(gray, sobel_kernel=11, thresh=(1.57 / 2.5, 1.58 / 1.5))

        combined = np.zeros_like(gradx_binary)
        combined[((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        # print(gradx_binary,grady_binary,mag_binary,dir_binary)
        # print("\ncombined",combined)
        combinedSchannel = np.zeros_like(combined)
        # combinedSchannel[((combined == 1) | ((imgSbin == 1)&(imgSbinH == 1)))] = 1
        combinedSchannel[(gradx_binary == 1) | ((imgSbin == 1) & (imgSbinH == 1))] = 1
        # print("\ncombinedSchannel",combinedSchannel)
        combinedSchannel = self.clipROI(combinedSchannel)
        return combinedSchannel

    def unwrap(self,img):
        src = np.float32(
            [[300, 700],
             [1095, 700],
             [612 - 122 - 2, 550],
             [676 + 165 + 2, 550]])
        srcC = src.copy()

        srcC[0, 0] = img.shape[1] / 2 - (src[1, 0] - src[0, 0]) / 2
        srcC[1, 0] = img.shape[1] / 2 + (src[1, 0] - src[0, 0]) / 2
        srcC[2, 0] = img.shape[1] / 2 - (src[3, 0] - src[2, 0]) / 2
        srcC[3, 0] = img.shape[1] / 2 + (src[3, 0] - src[2, 0]) / 2

        dst = np.float32(
            [[img.shape[1] / 2 - self.laneWidth / 2, 700],
             [img.shape[1] / 2 + self.laneWidth / 2, 700],
             [img.shape[1] / 2 - self.laneWidth / 2, 700 - self.laneHeight],
             [img.shape[1] / 2 + self.laneWidth / 2, 700 - self.laneHeight]])

        M = cv2.getPerspectiveTransform(srcC, dst)
        Minv = cv2.getPerspectiveTransform(dst, srcC)
        img_size = (img.shape[1], img.shape[0])
        # print("img_size",img_size)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped, Minv

    def computeHisto(self,img):
        histogram = np.sum(img[int(2 * img.shape[0] / 4):, :], axis=0)
        histoX = np.arange(0, img.shape[1])
        from scipy import signal

        peakind = signal.find_peaks_cwt(histogram, np.arange(1, histogram.shape[0]))
        # peak = signal.find_peaks_cwt(histogram, [100], max_distances=[100])
        # plutot qu'un seuil, il faudrait mieux conserver les 2 plus grands peaks
        minHistoThreshold = 10  # 15
        peakindMask = histogram[np.asarray(peakind)] > minHistoThreshold
        peakindFilter = histogram[peakind] > minHistoThreshold
        xMax = np.asarray(peakind)[peakindFilter]
        yMax = histogram[np.asarray(peakind)[peakindFilter]]
        # in peak, we will search peaks too clooser than a threshold
        # if find more than two peaks

        xMax2 = xMax
        yMax2 = yMax

        index = 0
        while (xMax2.shape[0] >= 2 and index < 10):
            index = index + 1
            toRemove = []
            xThreshold = 300
            for i in range(0, xMax2.shape[0] - 1):
                if (xMax2[i + 1] - xMax2[i] < xThreshold):
                    if (max(yMax2[i], yMax2[i + 1]) == yMax[i + 1]):
                        toRemove.append(i)
                    else:
                        toRemove.append(i + 1)

            xMax2 = np.delete(xMax2, toRemove)
            yMax2 = np.delete(yMax2, toRemove)

            # return(histoX,histogram,np.asarray(peakind)[peakindFilter],histogram[np.asarray(peakind)[peakindFilter]])

        return (histoX, histogram, xMax2, yMax2, xMax, yMax)

    def plotHisto(self,img, xHisto, yHisto, xMax2, yMax2, xMax, yMax):
        out_shape = (img.shape[0], img.shape[1], 3)
        outImg = np.zeros(out_shape, dtype=np.uint8)
        outImg[:, :, 2] = outImg[:, :, 1] = outImg[:, :, 0] = img * 255

        pts = np.array([np.transpose(np.vstack([xHisto, yHisto]))])
        cv2.polylines(outImg, pts, False, (0, 0, 255), 5)

        for i in range(0, xMax.shape[0]):
            cv2.circle(outImg, (xMax[i], yMax[i]), 5, (0, 255, 0), 5)

        for i in range(0, xMax2.shape[0]):
            cv2.circle(outImg, (xMax2[i], yMax2[i]), 5, (255, 0, 0), 5)

        return (outImg)

    def searchLines2(self,unwarpImgLines, xl, xr, linesParam):
        img = unwarpImgLines
        warp_zero = np.zeros_like(unwarpImgLines).astype(np.float)
        color_unwarp = np.dstack((warp_zero, warp_zero, unwarpImgLines * 255))

        nbSectY = 10
        widthSectorX = 200

        pixelsLY = np.array([])
        pixelsLX = np.array([])
        pixelsRY = np.array([])
        pixelsRX = np.array([])

        ySector = np.arange(img.shape[0], -1, -int(img.shape[0] / nbSectY))
        # print(xl,xr)

        for i in range(0, len(ySector) - 1):

            pixelsL = unwarpImgLines[ySector[i + 1]:ySector[i], int(xl - widthSectorX / 2):int(xl + widthSectorX / 2)]
            if (linesParam.lineL.lastLineOk):
                y = (ySector[i + 1] + ySector[i]) / 2
                x = linesParam.lineL.computeValueWithLastPoly(y)
                pixelsLpoly = unwarpImgLines[ySector[i + 1]:ySector[i],
                              int(x - widthSectorX / 2):int(x + widthSectorX / 2)]
                nbPixel = np.sum(pixelsL)
                nbPixelPoly = np.sum(pixelsLpoly)
                if (nbPixel < 2 or nbPixelPoly / nbPixel > 0.9):
                    pixelsL = pixelsLpoly
                    xl = x

            pixelsR = unwarpImgLines[ySector[i + 1]:ySector[i], int(xr - widthSectorX / 2):int(xr + widthSectorX / 2)]

            if (linesParam.lineR.lastLineOk):
                y = (ySector[i + 1] + ySector[i]) / 2
                x = linesParam.lineR.computeValueWithLastPoly(y)
                pixelsRpoly = unwarpImgLines[ySector[i + 1]:ySector[i],
                              int(x - widthSectorX / 2):int(x + widthSectorX / 2)]
                nbPixel = np.sum(pixelsR)
                nbPixelPoly = np.sum(pixelsRpoly)
                if (nbPixel < 2 or nbPixelPoly / nbPixel > 0.9):
                    pixelsR = pixelsRpoly
                    xr = x

            color_unwarp[ySector[i + 1]:ySector[i], int(xl - widthSectorX / 2):int(xl + widthSectorX / 2), 0] = 100
            color_unwarp[ySector[i + 1]:ySector[i], int(xr - widthSectorX / 2):int(xr + widthSectorX / 2), 1] = 100

            pixelsLY = np.concatenate([pixelsLY, ySector[i + 1] + np.where(pixelsL)[0]]);
            pixelsLX = np.concatenate([pixelsLX, xl - widthSectorX / 2 + np.where(pixelsL)[1]]);

            pixelsRY = np.concatenate([pixelsRY, ySector[i + 1] + np.where(pixelsR)[0]]);
            pixelsRX = np.concatenate([pixelsRX, xr - widthSectorX / 2 + np.where(pixelsR)[1]]);

            cv2.circle(color_unwarp, (int(xr), int((ySector[i + 1] + ySector[i]) / 2)), 10, (1, 1, 1), 4)
            cv2.circle(color_unwarp, (int(xl), int((ySector[i + 1] + ySector[i]) / 2)), 10, (1, 1, 1), 4)

            histogramR = np.sum(pixelsR, axis=0)
            if (len(histogramR) != 0):
                xr = xr - widthSectorX / 2 + np.mean(
                    np.where(histogramR == np.max(histogramR)))  # np.argmax(histogramR)

            histogramL = np.sum(pixelsL, axis=0)
            if (len(histogramL) != 0):
                xl = xl - widthSectorX / 2 + np.mean(
                    np.where(histogramL == np.max(histogramL)))  # np.argmax(histogramL)

        for k in range(0, pixelsRY.shape[0]):
            if (pixelsRX[k] < color_unwarp.shape[1] and pixelsRY[k] < color_unwarp.shape[0]):
                color_unwarp[int(pixelsRY[k]), int(pixelsRX[k]), 0] = 255
                color_unwarp[int(pixelsRY[k]), int(pixelsRX[k]), 1] = 255
                color_unwarp[int(pixelsRY[k]), int(pixelsRX[k]), 2] = 255

        for k in range(0, pixelsLY.shape[0]):
            if (pixelsLX[k] < color_unwarp.shape[1] and pixelsLY[k] < color_unwarp.shape[0]):
                color_unwarp[int(pixelsLY[k]), int(pixelsLX[k]), 0] = 200
                color_unwarp[int(pixelsLY[k]), int(pixelsLX[k]), 1] = 200
                color_unwarp[int(pixelsLY[k]), int(pixelsLX[k]), 2] = 200




                # print(ySector[i],xl,xr)

        # print(pixelsLY,pixelsRY)

        left_fit = np.polyfit(pixelsLY, pixelsLX, 2)
        left_fitx = left_fit[0] * pixelsLY ** 2 + left_fit[1] * pixelsLY + left_fit[2]

        right_fit = np.polyfit(pixelsRY, pixelsRX, 2)
        right_fitx = right_fit[0] * pixelsRY ** 2 + right_fit[1] * pixelsRY + right_fit[2]

        yLine = np.arange(0, unwarpImgLines.shape[0])
        left_fitxFull = left_fit[0] * yLine ** 2 + left_fit[1] * yLine + left_fit[2]
        right_fitxFull = right_fit[0] * yLine ** 2 + right_fit[1] * yLine + right_fit[2]

        pts = np.array([np.transpose(np.vstack([left_fitxFull.astype(np.int), yLine]))])
        # print(left_fitxFull,yLine)
        # print(left_fitxFull.shape,yLine.shape)
        cv2.polylines(color_unwarp, pts, False, (0, 255, 255), 5)
        pts = np.array([np.transpose(np.vstack([right_fitxFull.astype(np.int), yLine]))])
        cv2.polylines(color_unwarp, pts, False, (0, 255, 255), 5)

        y_evalL = np.max(pixelsLY)
        y_evalR = np.max(pixelsRY)

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / self.laneWidth  # meteres per pixel in x dimension

        left_fit_cr = np.polyfit(pixelsLY * ym_per_pix, pixelsLX * xm_per_pix, 2)
        right_fit_cr = np.polyfit(pixelsRY * ym_per_pix, pixelsRX * xm_per_pix, 2)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_evalL + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_evalR + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        left_pos = left_fitxFull[-1] * xm_per_pix - unwarpImgLines.shape[1] / 2 * xm_per_pix
        right_pos = right_fitxFull[-1] * xm_per_pix - unwarpImgLines.shape[1] / 2 * xm_per_pix

        return (color_unwarp, yLine, left_fitxFull, right_fitxFull, left_pos, right_pos, left_curverad, right_curverad,
                left_fit, right_fit)

    def backTransformation(self,img, Minv, yLine, left_fitxFull, right_fitxFull):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img[:, :, 0]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitxFull, yLine]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitxFull, yLine])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        return (result)

    # check if the 2 lines are acceptable
    def checkLines(self,left_fitxFull, right_fitxFull, left_pos, right_pos):
        # check lane width
        if (right_pos - left_pos > 4.5):
            return False
        if (right_pos - left_pos < 2.9):
            return False

        # check parallel Lines
        width = right_fitxFull[-1] - left_fitxFull[-1]
        widthMin = width * 0.7
        widthMax = width * 1.4

        for i in range(0, left_fitxFull.shape[0] - 1):
            if (right_fitxFull[i] - left_fitxFull[i] > widthMax):
                return False
            if (right_fitxFull[i] - left_fitxFull[i] < widthMin):
                return False

        return True

    def process_image(self,img):

        imgCorrected = self.calibrationCorrection(img)
        # imgLine,lineR,lineL = process_imageProject1(imgCorrected)
        # unwrapImag, Minv = unwrapWithLines(clipROI(imgLine),lineR,lineL)
        imgLinesExtracted = self.linesExtraction(imgCorrected)
        # unwarpImgLines, Minv = unwrapWithLines(imgLinesExtracted,lineR,lineL)
        unwarpImgLines, Minv = self.unwrap(imgLinesExtracted)

        xHisto, yHisto, xMax2, yMax2, xMax, yMax = self.computeHisto(unwarpImgLines)
        imgHisto = self.plotHisto(unwarpImgLines, xHisto, yHisto, xMax2, yMax2, xMax, yMax)
        if (xMax2.shape[0] == 2):
            self.xMaxLast = xMax2
        else:
            xMax2 = self.xMaxLast
        imgSearchLine = imgLine = imgLane = None

        if xMax2 is not None:
            imgSearchLine, yLine, left_fitxFull, right_fitxFull, left_pos, right_pos, left_curverad, right_curverad, left_fit, right_fit = self.searchLines2(
                unwarpImgLines, xMax2[0], xMax2[1], self.params)
            if (self.checkLines(left_fitxFull, right_fitxFull, left_pos, right_pos)):
                self.params.lineL.addPoly(left_fit)
                self.params.lineR.addPoly(right_fit)
                self.params.addCurvatureAndPosition(left_curverad, left_pos, right_curverad, right_pos)
            else:
                self.params.lineL.notDetected()
                self.params.lineR.notDetected()

            left_fitMean = self.params.lineL.computeValue(yLine)
            right_fitMean = self.params.lineR.computeValue(yLine)
            imgLane = self.backTransformation(imgCorrected, Minv, yLine, left_fitMean, right_fitMean)
            curvature, position = self.params.computeCurvatureAndPosition()
            strC = "Curvature : %0.2fm" % (curvature)
            strP = "Position : %0.2fm" % (position)
            cv2.putText(imgLane, strC, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(imgLane, strP, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            # print(left_fit)
            # print(params.lineL.polyMean())
        fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX;
        fontScale = 1;

        strL = "Left poly : %0.2e %0.2e %0.2e, curv %4.1f" % (left_fit[0], left_fit[1], left_fit[2], left_curverad)
        strR = "Right poly: %0.2e %0.2e %0.2e, curv %4.1f" % (right_fit[0], right_fit[1], right_fit[2], right_curverad)
        cv2.putText(imgSearchLine, strL, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(imgSearchLine, strR, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        #imageDebug = debug_image2(imgLane, imgHisto, imgSearchLine, unwarpImgLines, imgLinesExtracted, imgSearchLine,
        #                          imgLine)

        return imgLane,imgHisto,imgSearchLine,unwarpImgLines, imgLinesExtracted,imgLine  # just the lane detection image (same format)
        #return imageDebug  # debug image with intermediate processing steps

    def run(self):
        self.loadCalibration()
        image = imread('test_images/test3.jpg')
        imgLane, imgHisto, imgSearchLine, unwarpImgLines, imgLinesExtracted, imgLine  = self.process_image(image)
        plt.figure()
        plt.imshow(imgLane)
        plt.figure()
        plt.imshow(imgHisto)
        plt.figure()
        plt.imshow(imgSearchLine)
        plt.figure()
        plt.imshow(unwarpImgLines)
        plt.figure()
        plt.imshow(imgLinesExtracted)
        #plt.figure()
        #plt.imshow(imgLine)
        plt.show()

if __name__ == "__main__":
    obj = LaneDetection()
    #obj.cameraCalibration()
    obj.run();

