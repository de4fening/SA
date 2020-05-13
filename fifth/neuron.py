import numpy

class Neuron:
    weight = ()
    dimension = 0
    points = []
    __rCritical = 100
    __isCritical = False
    __betaCoeff = 3

    def __init__(self, weight, speedCoeff = 0.2, critical = False, rCritical = 100, betaCoeff = 3, init = False):
        self.weight = weight
        self.dimension = len(weight)
        self.speedCoeff = speedCoeff
        self.__rCritical = rCritical
        self.__isCritical = critical
        self.__betaCoeff = betaCoeff
        self.points = []
        if init:
            self.points.append(weight)

    def distance(self, point):
        if self.dimension != len(point):
            raise AttributeError("dimensions aren't equals")
        return numpy.sqrt(sum([(point[i] - self.weight[i]) ** 2 for i in range(0, self.dimension)]))

    def setSpeedCoef(self, newCoeff):
        self.speedCoeff = newCoeff

    def refresh(self, point, isWin = False):
        if self.dimension != len(point):
            raise AttributeError("dimensions aren't equals")
        if isWin:
            self.points.append(point)
        if self.__isCritical:
            coef = self.speedCoeff
            if not isWin:
                coef = self.speedCoeff * (1 - 1 / (1 + numpy.e ** (-self.__betaCoeff * (self.distance(point) - self.__rCritical))))
            self.weight = tuple([self.weight[i] + coef * (point[i] - self.weight[i]) for i in range(0, self.dimension)])
        else:
            if isWin:
                coef = self.speedCoeff
                self.weight = tuple([self.weight[i] + coef * (point[i] - self.weight[i]) for i in range(0, self.dimension)])

    def getPoints(self):
        return self.points