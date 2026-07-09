import matplotlib.pyplot as plt
import numpy as np
from data import DatasetSample
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from task import Equation, Task

def drawLine(axis: Axes, equation: Equation, sample: DatasetSample, lambdas: NDArray, color: str | tuple[float, ...] = "green"):
    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)
    
    X, Y = np.meshgrid(x, y)
    Z = sum(equation(np.array([X,Y]), sample.parameters, sample.saddle_point, sample.eigens, lambdas)[0])

    axis.contour(X, Y, Z, levels=[0], colors=color)

def drawExample(task: Task, equations: list[Equation], dataset: list[DatasetSample], lambdas: NDArray) -> tuple[Figure, Axes]:
    figure, axis = plt.subplots()
    axis.set_xlim(0,1)
    axis.set_ylim(0,1)
    
    if task.dimensions != 2:
        raise ValueError(f"This module intended only for 2D tasks! This task is {task.dimensions}D!")

    # Create dot array
    for row in dataset:
        x, y = row.start
        indicator = row.indicator
        marker = "+"
        if indicator > 0:
            color = "red"
        elif indicator < 0:
            color = "blue"
        else:
            color = "gray"
        axis.plot(x, y, color = color, marker = marker)

    color_map = plt.get_cmap("tab10")
    allowed_indices = [1, 2, 4, 5, 6, 7, 8, 9]

    # Create all specified lines
    row = dataset[0]
    for i, equation in enumerate(equations):
        color_idx = allowed_indices[i % len(allowed_indices)]
        line_color = color_map(color_idx)
        drawLine(axis, equation, row, lambdas, color = line_color) 
        func_name = equation.__name__
        axis.plot([], [], color=line_color, label=func_name)
    
    return figure, axis
