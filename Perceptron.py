from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import Bunch

class Perceptron:
    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data
        #segregating one class from the others, if target == 0, the new value is 0, if no is 1
        self.y = np.where(iris.target == 0, 0, 1)
    def main(self):
        xValidate,xTrain,yValidade,yTrain = self.trainTestSplit(self.X, self.y, 0.2)

        print(yTrain)
        print(yValidade)

    def split_Dataset(self):
        print("Values")
    def train(self):
        t = 25
        print()
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
        print(y[permutation])
        return  xValidate, xTrain,yValidate,yTrain



teste = Perceptron()
teste.main()
