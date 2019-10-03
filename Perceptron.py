from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.utils import Bunch
# Só pra testar
class Perceptron:
    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data
        #segregating one class from the others, if target == 0, the new value is 0, if no is 1
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
        # print(yValidade)
        print("Media de acurracy is: ", acurracy/10)
    def split_Dataset(self):
        print("Values")
    def addX1(self, xTrain):
        size = len(xTrain)
        ones = - np.ones((size, 1))
        X = np.concatenate((ones, xTrain), axis=1)
        return X

    def train(self, xTrain, dtrain, w, seasons=2, learningRate= 0.01):
        num = 0
        while True:
            error = False
            for i in range(len(xTrain)):
                u = self.summationValuesMatrix(xTrain[i], w)
                y = 0
                if u <= 0:
                    y = 0
                else:
                    y = 1

                if y != dtrain[i]:
                    err = dtrain[i] - y
                    for j in range(len(xTrain[i])-1):
                        w[j] = w[j]+(learningRate * err * xTrain[i][j])
                    error = True
            num += 1
            if num > seasons or not error:
                break
        return w
    def summationValuesMatrix(self, x,w):
        valueMultiply = 0
        for i in range(0, len(x)-1):
            valueMultiply += x[i] * w[i]
        return valueMultiply

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
teste = Perceptron()
teste.main()
