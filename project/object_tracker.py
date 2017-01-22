
# Vehicles tracking implementation with a GNN algorithm (Global nearest neighbor  http://www.control.isy.liu.se/student/graduate/TargetTracking/Lecture5.pdf).


import cv2
import math
import numpy as np

# objet t track class
class Object:
    def __init__(self,_position, _bBox, _targetIndex):
        self.position = _position
        self.age = 1
        self.targetIndex = _targetIndex
        self.distanceThreshold = 50;
        self.ageThreshold = 5;

        self.MeanFrameNumber = 10
        self.listBoundingBox =[]
        self.listPosition = []
        self.listPosition.append(_position)
        self.listBoundingBox.append(_bBox)

    # compute distance between this and targe
    def distance(self,_position):
        return math.sqrt(math.pow(self.position[0]-_position[0],2)+math.pow(self.position[1]-_position[1],2));

    # update object from target list
    def update(self,targets,bBox):
        minDistance = float("inf");
        for idxTarget in range(0,len(targets)):
            dist = self.distance(targets[idxTarget])
            if(minDistance>dist):
                minDistance = dist
                idxMin = idxTarget

        if(minDistance<self.distanceThreshold):
            self.targetIndex = idxMin
            self.position = targets[self.targetIndex]
            self.listBoundingBox.append(bBox[self.targetIndex])
            if(len(self.listBoundingBox)>self.MeanFrameNumber):
                self.listBoundingBox.pop(0)
            self.listPosition.append(self.position)
            if (len(self.listPosition) > self.MeanFrameNumber):
                self.listPosition.pop(0)
            self.age+=1
            if(self.age>self.ageThreshold*2):
                self.age = self.ageThreshold*2
        else:
            self.age -= 1
            if (self.age < 0):
                self.age = 0
            self.targetIndex = -1

    # compute the average position and bouding box
    def getPosition(self):
        posMean=np.mean(np.array(self.listPosition), axis=0).astype(np.uint16)
        bbMean=np.mean(np.array(self.listBoundingBox), axis=0).astype(np.uint16)
        return posMean,bbMean

# vehicule, derivate from object
class Vehicle(Object):
     def __init__(self, position,_bBox, _targetIndex):
        Object.__init__(self, position,_bBox, _targetIndex)

# tracker class, to manage vehicles list
class Tracker:
    def __init__(self):
        self.cars=[]

    # new targets list from svc detection
    def newDetection(self,targets,bBoxs):
        # if no target, decrease of every cars
        if(len(targets)==0):
            for car in self.cars:
                car.age-=1
                if(car.age==0):
                    self.cars.remove(car)
            return
        # if no car, add all
        if(len(self.cars)==0):
            for i in range (0, len(targets)):
                self.cars.append(Vehicle(targets[i],bBoxs[i],i))

        else:
            # for each car update position from target
            for car in self.cars:
                car.update(targets,bBoxs)
            # remove car with 0 age
            toRemove = []
            for car in self.cars:
                if(car.age==0):
                    toRemove.append(car)
            # remove age = 0
            self.cars = [i for j, i in enumerate(self.cars) if j not in toRemove]

            # search for car with same target
            for i in range(0,len(targets)):
                t=[]
                for car in self.cars:
                    if(car.targetIndex == i):
                        t.append(car)
                while(len(t)>1): # faut supprimer quelque chose
                    if(t[0].age<t[1].age):
                        self.cars.remove(t[0])
                        t.pop(0)
                    else:
                        self.cars.remove(t[1])
                        t.pop(1)
            # create new vehicule for target not allocated
            for i in range(0,len(targets)):
                findIt = False
                for car in self.cars:
                    if(car.targetIndex==i):
                        findIt=True
                if(not findIt):
                    self.cars.append(Vehicle(targets[i], bBoxs[i], i))


    # get tracker results
    def getTrackingResult(self):
        pos = []
        bBox = []
        for car in self.cars:
            if car.age>=car.ageThreshold:
                p,bb=car.getPosition()
                pos.append(p)
                bBox.append(bb)

        return pos,bBox