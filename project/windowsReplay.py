import pickle
import cv2

class windowsReplay:
    def __init__(self):
        self.exportVideo = []

    def load(self,file):
        with open(file, 'rb') as handle:
            self.exportVideo = pickle.load(handle)


    def run(self):
        print(cv2.__version__)
        self.load("windows01.pkl")
        print("nb frame:",len(self.exportVideo))


if __name__ == "__main__":
    obj = windowsReplay()
    obj.run()
