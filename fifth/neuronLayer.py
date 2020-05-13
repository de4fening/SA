import neuron as Neu
import random as rnd
import numpy as np


def random_weight(weight):
    a = tuple([rnd.uniform(0.5 - 1 / np.sqrt(weight), 0.5 + 1 / np.sqrt(weight)) for _ in range(0, weight)])
    return a

class NeyronLayer:
    neuronList = []
    size = 0
    __dimension = 0
    __isSelfOrganized = False
    __isModified = False
    __rCritical = False
    __modCritical = 0
    __speedCoeff = 0

    def __init__(self, size = 4, weight = 5, speedCoeff = 0.2, selfOrganized = False, layerCritical = 100, modCritical = 100, modified = False):
        self.size = size
        self.__speedCoeff = speedCoeff
        self.__modCritical = modCritical
        self.__rCritical = layerCritical
        self.__dimension = weight
        self.__isModified = modified
        self.__isSelfOrganized = selfOrganized
        for i in range(0, size):
            self.neuronList.append(Neu.Neuron(random_weight(weight), speedCoeff, modified, modCritical, 3, False))

    def addRes(self, point):
        if self.__dimension != len(point):
            raise AttributeError("dimensions are not equals")

        result = [neuron.distance(point) for neuron in self.neuronList]
        minim = result[0]
        minimIndex = 0
        for i in range(0, self.size):
            if (result[i] < minim):
                minim = result[i]
                minimIndex = i

        if self.__isSelfOrganized and minim > self.__rCritical:
            self.size += 1
            self.neuronList.append(Neu.Neuron(random_weight(self.__dimension), self.__speedCoeff, self.__isModified, self.__modCritical, 3, True))
        else:
            for i in range(0, self.size):
                self.neuronList[i].refresh(point, i == minimIndex)

    def showResults(self):
        for i in range(0, self.size):
            print("\n\n-------------------" + str(i) + "------------------\n" )
            for point in self.neuronList[i].getPoints():
                print(str(point) + "\n")