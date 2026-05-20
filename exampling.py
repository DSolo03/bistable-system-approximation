from pylab import plot,show,contour,xlim,ylim,legend,savefig,clf
import numpy as np

from task import Task

def drawLine(equation:Task.expansion, sample, lambdas=None, color="green"):
    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)
    
    X, Y = np.meshgrid(x, y)
    Z = sum(equation([X,Y],sample["parameters"], sample["saddlePoint"], sample["eigens"], lambdas)[0])
    
    contour(X, Y, Z, levels=[0], colors=color)

def drawExample(task: Task, dataset, lambdas):
    xlim([0,1])
    ylim([0,1])
    
    # Create dot array
    for row in dataset:
        x, y = row["start"]
        indicator = row["indicator"]
        marker = "+"
        if indicator > 0:
            color = "red"
        elif indicator < 0:
            color = "blue"
        else:
            color = "gray"
        plot(x, y, color = color, marker = marker)

    # Create saddle point and saddle separatrix
    row = dataset[0]
    plot(*row["saddlePoint"], color = "green", marker = "o")
    drawLine(task.saddleAlphaSeparatrix, row, [], "orange")
    drawLine(task.saddleOmegaSeparatrix, row, [], "green")
    drawLine(task.RGR,row,[],"black")
    # Create separator
    row = dataset[0]
    drawLine(task.expansion, row, lambdas, "red")

    show()

