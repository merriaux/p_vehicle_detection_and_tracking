import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob


class LoadData:
    def __init__(self):
        self.notcars = []
        self.cars = []




    def loadSmallDataSet(self):
         # Divide up into cars and notcars
         images = glob.iglob('../smallDataset/**/*.jpeg', recursive=True)
         self.cars = []
         self.notcars = []
         for image in images:
             if 'image' in image or 'extra' in image:
                 self.notcars.append(image)
             else:
                 self.cars.append(image)



    def loadDataset(self):
        images = glob.iglob('../dataset/vehicles/**/*.png', recursive=True)
        for image in images:
            self.cars.append(image)

        images = glob.iglob('../dataset/non-vehicles/**/*.png', recursive=True)
        for image in images:
            self.notcars.append(image)


    # Define a function to return some characteristics of the dataset
    def data_look(self,car_list, notcar_list):
        data_dict = {}
        # Define a key in data_dict "n_cars" and store the number of car images
        data_dict["n_cars"] = len(car_list)
        # Define a key "n_notcars" and store the number of notcar images
        data_dict["n_notcars"] = len(notcar_list)
        # Read in a test image, either car or notcar
        example_img = mpimg.imread(car_list[0])
        # Define a key "image_shape" and store the test image shape 3-tuple
        data_dict["image_shape"] = example_img.shape
        # Define a key "data_type" and store the data type of the test image.
        data_dict["data_type"] = example_img.dtype
        # Return data_dict
        return data_dict

    def printdatasetInfo(self):
        data_info = self.data_look(self.cars, self.notcars)
        print(data_info)

    def displayTwoRandomImages(self):
        car_ind = np.random.randint(0, len(self.cars))
        notcar_ind = np.random.randint(0, len(self.notcars))

        # Read in car / not-car images
        car_image = mpimg.imread(self.cars[car_ind])
        notcar_image = mpimg.imread(self.notcars[notcar_ind])
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(car_image)
        plt.title('Example Car Image\n' + self.cars[car_ind])
        plt.subplot(122)
        plt.imshow(notcar_image)
        plt.title('Example Not-car Image\n' + self.notcars[notcar_ind])
        plt.show()

    def run(self):
        #self.loadSmallDataSet()
        self.loadDataset()
        self.printdatasetInfo()
        self.displayTwoRandomImages()

if __name__ == "__main__":
    obj = LoadData()
    obj.run()
