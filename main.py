import data
from task import TwoDTask
import seaborn as sns
import pprint
import matplotlib.pyplot as plt
import pandas as pd

from sys import exit

def heatmap(task, dataset:dict):
    a,b=ml.prepareFeatures(task,dataset)
    corr = a.corr()
    sns.heatmap(corr,annot=True,cmap="coolwarm")
    plt.show()

if __name__ == '__main__':
    import ml
    import exampling
    import utils
    task = TwoDTask()

    dataset = utils.loadDataset()
    #dataset = data.generateDataset(task,5000)
    #utils.saveDataset(dataset)
    weights, statistics = ml.train(task, dataset)
    pprint.pprint(weights)
    pprint.pprint(statistics)
    example = data.generateExample(task)
    pprint.pprint(example[0])
    exampling.drawExample(task, example, weights)