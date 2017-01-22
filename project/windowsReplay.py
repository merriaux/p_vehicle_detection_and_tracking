import pickle
import cv2

# To experimente the tracking algorithm, I have recorded all the sliding windows predited car for each video frame in pickle file, and use it to tune tracking parameters.


class WindowsReplay:
    def __init__(self):
        self.exportVideo = []
    # load of the windows and bouding box for each frame
    def load(self,file):
        with open(file, 'rb') as handle:
            self.exportVideo = pickle.load(handle)


    def run(self):
        print(cv2.__version__)
        self.load("windows01.pkl")
        print("nb frame:",len(self.exportVideo))


if __name__ == "__main__":
    obj = WindowsReplay()
    obj.run()
