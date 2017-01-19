import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

from loadData import LoadData
from features import Features

class Classifier:
    def __init__(self):
        self.svc = LinearSVC(C=0.0001)


    def train(self,x,y):
        self.svc.fit(x,y)

    def eval(self,x,y):
        return self.svc.score(x,y)


    def predict(self,x):
        return self.svc.predict(x)


    def save(self,file):
        joblib.dump(self.svc, file)

    def load(self,file):
        self.svc = clf = joblib.load(file)

    def confusionMatrix(self,X,y_true):
        y_pred = self.predict(X)
        return confusion_matrix(y_true, y_pred)

    def run(self):
        t = time.time()
        feat = Features()
        feat.loadFromPickle("features01.pkl")
        print("feature shape", feat.X_train.shape)
        t2 = time.time()
        print(t2 - t, 'Seconds to load features.')

        t = time.time()
        self.train(feat.X_train,feat.y_train)
        t2 = time.time()
        print(t2 - t, 'Seconds to train classifier.')

        t = time.time()
        print("accuracy on train", self.svc.score(feat.X_train, feat.y_train))
        print("accuracy on test", self.svc.score(feat.X_test, feat.y_test))
        t2 = time.time()
        print(t2 - t, 'Seconds to test classifier.')

        self.save("svcModel01.pkl")



    def runConfusionMatrix(self):
        feat = Features()
        feat.loadFromPickle("features01.pkl")
        self.load("svcModel01.pkl")
        t = time.time()
        print("accuracy on train", self.svc.score(feat.X_train, feat.y_train))
        print("accuracy on test", self.svc.score(feat.X_test, feat.y_test))
        t2 = time.time()
        print(t2 - t, 'Seconds to test classifier.')
        print("confusion matrix Train: \n",self.confusionMatrix(feat.X_train,feat.y_train))
        print("confusion matrix Test: \n", self.confusionMatrix(feat.X_test, feat.y_test))
        print(self.svc.coef_)
        print(self.svc.coef_.shape)
        print(feat.X_test.shape)
        val=np.sort(np.abs(self.svc.coef_))
        id = np.argsort(np.abs(self.svc.coef_))
        print(val, id)
        print(val.shape,id.shape)
        plt.figure()
        plt.plot(val[0,:])
        plt.title('valeur coef')
        plt.figure()
        plt.plot(id[0,:],'.')
        plt.title('index coef')
        plt.show()

    def runTestClassif(self):
        feat = Features()
        feat.loadFromPickle("features01.pkl")
        self.load("svcModel01.pkl")
        print("accuracy on test", self.svc.score(feat.X_test, feat.y_test))
        y=self.predict(feat.X_test)
        print("y.shape",y.shape,'y=',np.sum(y))
        print('yPred!=y', np.sum(y!=feat.y_test))


if __name__ == "__main__":
    obj = Classifier()
    obj.run()
    obj.runConfusionMatrix()
    #obj.runTestClassif()