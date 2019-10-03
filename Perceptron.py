from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import Bunch

class Perceptron:
    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris
        #segregating one class from the others, if target == 0, the new value is 0, if no is 1
        self.y = np.where(iris.target == 0, 0, 1)
    def main(self):
        print(self.X.target)
        #print(self.y)
    def split_Dataset(self):
        print("Values")
    def train(self):
        t = 25
        print()


teste = Perceptron()
teste.main()
