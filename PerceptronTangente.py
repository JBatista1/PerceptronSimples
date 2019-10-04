from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import random

from matplotlib.colors import ListedColormap
class PerceptronTangente:
    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data
        self.D = np.where(iris.target == 2, 0, 1)
        self.w = []
        for i in range(0, 5):
            if i == 0 :
                self.w.append(0)
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

    def train(self, xTrain, dtrain, w, seasons=1000, learningRate= 0.01):
        num = 0
        while True:
            error = False
            for i in range(len(xTrain)):
                u = self.summationValuesMatrix(xTrain[i], w)
                y = (1 - np.exp(-u)) / (1 + np.exp(-u))
                if y != dtrain[i]:
                    err = dtrain[i] - y
                    yline = 0.5 * (1 - (y*y))
                    for j in range(len(xTrain[i])-1):
                        w[j] = w[j]+(learningRate * err * yline * xTrain[i][j])
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

    def plot_decision_regions(X, y, classifier, resolution=0.02):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)
teste = PerceptronTangente()
teste.main()
