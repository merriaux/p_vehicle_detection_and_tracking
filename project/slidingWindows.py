import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#Sliding windows implementation, with region of interest in the image.
#Pyramid windows, with size step and overlap tuning parameters.
#Extraction and resizing of all image windows for prediction in classif class.

class SlidingWindows:
    def __init__(self):
        self.window_list = [];

    #  draw_boxes function from windows list
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    # draw_boxes function from windows numpy array
    def draw_boxesNP(self, img, bboxes, color=(0, 0, 255), thick=6):
            # Make a copy of the image
            imcopy = np.copy(img)
            # Iterate through the bounding boxes
            for i in range(0,bboxes.shape[0]):
                # Draw a rectangle given bbox coordinates
                cv2.rectangle(imcopy, (bboxes[i,0],bboxes[i,1]), (bboxes[i,2],bboxes[i,3]), color, thick)
            # Return the image copy with boxes drawn
            return imcopy

    # slide windows function
    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan / nx_pix_per_step)-1
        ny_windows = np.int(yspan / ny_pix_per_step)-1

        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                w=((startx, starty), (endx, endy))
                if(self.isWindowsInROI(w)):
                    window_list.append(w)
        # Return the list of windows
        return window_list

    # init ROI from project 1 and 4
    def initROI(self,img):
        imshape = img.shape
        imgROI = np.copy(img)
        vertices = np.array([[(0 + 200, imshape[0] - 40), (138, 450), (540, 350),
                              (imshape[1] - 540, 350), (imshape[1] - 0, 350), (imshape[1] - 0, imshape[0] - 40)]],dtype=np.int32)
        self.maskROI = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(self.maskROI, vertices, 255)
        cv2.polylines(imgROI, vertices, True, (0, 255, 0), 3)
        return(imgROI)

    # test if the windows is in ROI from list of coordonates
    def isWindowsInROI(self,window):
        try:
            test=not ((self.maskROI[window[0][1], window[0][0]] == 0) and (self.maskROI[window[0][1], window[1][0]-1] == 0) and
                 (self.maskROI[window[1][1]-1, window[0][0]] == 0) and (self.maskROI[window[1][1]-1, window[1][0]-1] == 0))
        except:
            #print(window)
            test=False

        return(test)

    # test if the windows is in ROI from a numpy array
    def isWindowsInRoiNP(self,window):
        try:
            test=not ((self.maskROI[window[1], window[0]] == 0) and (self.maskROI[window[1], window[2]-1] == 0) and
                 (self.maskROI[window[3]-1, window[0]] == 0) and (self.maskROI[window[3]-1, window[2]-1] == 0))
        except:
            #print(window)
            test=False

        return(test)

    # call slide_windows for different windows size
    def pyramid_windows(self,img,windows_size=(32,128),overlap=0.75):
        self.initROI(img)
        window_list = []
        window_size = windows_size[0]
        while (window_size<=windows_size[1]):
            window_list += self.slide_window( img, xy_window=(window_size, window_size), y_start_stop=[2*128, img.shape[0]], xy_overlap=(overlap, overlap))
            window_size = window_size + windows_size[0]
        return(window_list)

    # extract small img form windows coordonates
    def windowsImgExtract(self,img,window):
        return img[window[0][1]:window[1][1],window[0][0]:window[1][0],:]

    # resize small image to 64x64 pixels
    def imgResize(self,img):
        return cv2.resize(img,(64,64),interpolation = cv2.INTER_CUBIC)


    def run(self):
        plt.close("all")
        image = cv2.imread('test_images/test1.jpg')
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.initROI(image)

        windows = self.slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
        print("Windows numbers:",len(windows))

        window_img = self.draw_boxes(image, windows, color=(0, 0, 255), thick=6)
        plt.imshow(window_img)
        #plt.show()
        plt.figure()

        windows = self.pyramid_windows(image,windows_size=(32,128),overlap=0.5)
        print("Windows pyramide numbers:", len(windows))

        window_img = self.draw_boxes(image, windows, color=(0, 0, 255), thick=3)
        plt.imshow(window_img)
        plt.figure()
        window_ind = np.random.randint(0, len(windows))
        win =[]
        win.append(windows[window_ind])
        window_img = self.draw_boxes(image, win, color=(0, 255, 255), thick=6)
        plt.imshow(window_img)
        plt.figure()
        plt.imshow(self.imgResize(self.windowsImgExtract(window_img,windows[window_ind])))

        plt.show()





if __name__ == "__main__":
    obj = SlidingWindows()
    obj.run()
