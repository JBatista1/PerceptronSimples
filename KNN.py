
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import random
import math
import operator
class KNN :
    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data
        # segregating one class from the others, if target == 0, the new value is 0, if no is 1
        self.D = iris.target
        self.names = iris.target_names

    def main(self):
        mediumAcurracy = 0
        for i in range(10):
            dataset = self.concatenedDataSetName()
            xValidate, xTrain, = self.trainTestSplit(dataset, 0.2)
            prediction = []
            distance = 0
            k = 3
            for i in range(len(xValidate)):
                neighbors = self.getNeighbors(xTrain,xValidate[i],k)

                result = self.getResponse(neighbors)
                prediction.append(result)
            acurracy = self.getAccuracy(xValidate, prediction)
            mediumAcurracy += acurracy
        print("Acurracy medium is:", repr(mediumAcurracy/10))

    def concatenedDataSetName(self):
        dataset = []
        lengrh = len(self.X)
        values = []

        for i in range(lengrh):
            if self.D[i] == 0:
                dataset.append(self.names[0])
            elif self.D[i] == 1:
                dataset.append(self.names[1])
            else:
                dataset.append(self.names[2])
        for i in range(lengrh):
            temp = [1,1,1,1,'a']
            for j in range(len(self.X[0])+1):
                if j == len(self.X[0]):

                    temp[j] = dataset[i]
                else:
                    temp[j] = self.X[i][j]

            values.append(temp)
        return values

    def trainTestSplit(self,X,test_size):
        random.shuffle(X)
        #value for create array training
        numberElementsForTraining = len(X) * test_size
        #Array permutation all elements in dataset
        permutation = np.random.permutation(len(X))
        # convert to int because func in array only accepts int
        numberElementsForTraining = int(numberElementsForTraining)
        #Crop arrays based in value for train
        xValidate = X[:numberElementsForTraining]
        xTrain = X[numberElementsForTraining:]
        return xValidate, xTrain

    def euclideanDistance(self, values1, values2):
        distance = 0
        for i in range(len(values1)-1):
            distance += pow((float(values1[i]) - float(values2[i])),2)
        distance = math.sqrt(distance)
        return distance

    def getNeighbors(self, trainingDataSet, testIntance, k):
        distance = []
        lengrh = len(testIntance)-1
        for i in range(len(trainingDataSet)):
            dist = self.euclideanDistance(testIntance, trainingDataSet[i])
            distance.append((trainingDataSet[i], dist))
        distance.sort(key = operator.itemgetter(1))
        neighbors = []

        for i in range(k):
            neighbors.append(distance[i][0])
        return neighbors
    def getResponse(self,neighbors) :
        classVotes = {}
        for i in range(len(neighbors)):
            response = neighbors[i][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key= operator.itemgetter(1), reverse = True)
        return sortedVotes[0][0]
    def getAccuracy(self, testSet, predications):
        correct = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predications[i]:
                correct += 1

        return (correct/float(len(testSet))) * 100.0
    # def manhattan(self, point1, point2):


knnTest = KNN()
knnTest.main()
