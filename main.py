import data
import matplotlib.pyplot as plt
from task import TwoDTask

if __name__ == '__main__':
    import ml
    import exampling
    task = TwoDTask()
    
    dataset = data.generate_dataset(task,size = 5000)
    weights, statistics = ml.train(task, dataset)

    print(f"[Result] Weights:\n{weights}")
    print(f"[Result] Statistics:\n{statistics}")

    example = data.generate_example(task, 100)
    figure, axis = exampling.draw_example(task, [task.expansion, task.saddle_omega_separatrix, task.RGR], example, weights)
    axis.legend()
    plt.show()
    