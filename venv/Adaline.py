from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import random

from matplotlib.colors import ListedColormap
# Só pra testar
class Adaline:
    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data
        self.D = np.where(iris.target == 2, 0, 1)
        self.w = []
        for i in range(0, 5):
            if i == 0 :
                self.w.append(-1)
            else:
                self.w.append(random.random())
        self.w = np.array(self.w)

    def main(self):
        acurracy = 0
        for i in range(10):
            xValidate,xTrain,dValidade,dTrain = self.trainTestSplit(self.X, self.D, 0.2)
            xTrain = self.addX1(xTrain)
            xValidate = self.addX1(xValidate)
            self.w = self.train(xTrain, dTrain, self.w)
            acurracy += self.validate(xValidate,dValidade)

        print("Media de acurracy is: ", acurracy/10)


    def addX1(self, xTrain):
        size = len(xTrain)
        ones = - np.ones((size, 1))
        X = np.concatenate((ones, xTrain), axis=1)
        return X

    def quadraticError(self, X, y, w):
        p = len(X)
        error = 0.0
        for i in range(0, len(X)):
            u = self.summationValuesMatrix(X[i], w)
            aux = y[i] - u
            error += pow(aux, 2)
        error /= p
        return error

    def train(self, xTrain, dtrain, w, seasons=1000, learningRate = 0.0001):
        num = 0
        for epoch in range(seasons):
            previousError = self.quadraticError(xTrain,dtrain, w)

            for i in range(len(xTrain)):
                u = self.summationValuesMatrix(xTrain[i], w)
                e = learningRate*(dtrain[i]-u)
                aux = self.multiplyValueForMatrix(xTrain[i], e)
                w = self.sumMatrix(w, aux)
            currentError = self.quadraticError(xTrain,dtrain, w)
            error = currentError - previousError

            if abs(error) < 0.000001 :
                return w

        return w
    def summationValuesMatrix(self, x,w):
        valueMultiply = 0

        for i in range(len(x)):

            valueMultiply += x[i] * w[i]
        return valueMultiply

    def sumMatrix(self, matrix1, matrix2):
        for i in range(len(matrix1)):
            matrix1[i] += matrix2[i]
        return  matrix1

    def multiplyValueForMatrix(self, matrix, value):
        for i in range(len(matrix)):
            matrix[i] = matrix[i]*value
        return matrix
    def trainTestSplit(self,X,y,test_size):
        #value for create array training
        numberElementsForTraining = len(y) * test_size

        #Array permutation all elements in dataset
        permutation = np.random.permutation(len(y))
        # convert to int because func in array only accepts int
        numberElementsForTraining = int(numberElementsForTraining)

        #Crop arrays based in value for train
        xValidate = X[permutation[:numberElementsForTraining]]
        xTrain = X[permutation[numberElementsForTraining:]]
        yValidate = y[permutation[:numberElementsForTraining]]
        yTrain = y[permutation[numberElementsForTraining:]]
        return  xValidate, xTrain,yValidate,yTrain

    def validate(self,xValidate, dValidate):
        positive = 0
        for index in range(0, len(xValidate)):
            u = self.summationValuesMatrix(xValidate[index], self.w)
            y = 0
            if u <= 0:
                y = 0
            else:
                y = 1
            if y == dValidate[index]:
                positive += 1
        return (positive/len(dValidate))



teste = Adaline()
teste.main()
