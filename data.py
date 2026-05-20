from multiprocessing import Pool
import numpy as np
from progress.bar import IncrementalBar
import scipy.integrate as integrate
from sklearn.preprocessing import normalize

from task import Task, Parameters

def checkSample(task: Task, start: list, parameters: Parameters, time:float = 10000, step:float = 0.01, method:str = "Radau") -> int:
    func = lambda _,z: task.system(z, parameters)
    solution = integrate.solve_ivp(func, [0,time], start, t_eval=np.arange(0, time, step), method=method, first_step=step)

    return task.classifyPoint(solution.y[:,-1])

def createDatasetSample(task: Task):
    while True: 
        start, parameters = task.sample()

        if not task.multistableCondition(parameters):
            continue   

        saddlePoint, eigens = task.getSaddle(parameters)
        if saddlePoint is None or eigens is None:
            continue

        indicator = checkSample(task, start, parameters)
        
        if indicator == 0:
            continue

        return {"start":start, "parameters":parameters, "saddlePoint":saddlePoint, "eigens":eigens, "indicator":indicator}

def generateDataset(task: Task, size: int = 1000, processes:int = 6):
    bar = IncrementalBar('[Dataset] Generating', max = size)
    dataset = []

    def callback(sample):
        dataset.append(sample)
        bar.next()

    createDatasetSample(task)

    print(f"[Dataset] Creating dataset of size {size} samples, on {processes} processes..")  
    with Pool(processes) as pool:
        jobs = []
        for _ in range(0, size):
            job = pool.apply_async(createDatasetSample, args=(task,), callback=callback)
            jobs.append(job)
        [job.wait() for job in jobs]
    bar.finish()

    return dataset

def createExampleSample(task: Task, parameters: Parameters, saddlePoint: np.array, eigens: np.array):
    start,_ = task.sample()
    indicator = checkSample(task, start, parameters)

    return {"start":start, "parameters":parameters, "saddlePoint":saddlePoint, "eigens":eigens, "indicator":indicator}

def generateExample(task: Task, size: int = 200, processes:int = 6):
    bar = IncrementalBar('[Example] Generating', max = size)
    dataset = []

    def callback(sample):
        dataset.append(sample)
        bar.next()

    sample = createDatasetSample(task)

    print(f"[Example] Creating dataset of size {size} samples, on {processes} processes..")  
    with Pool(processes) as pool:
        jobs = []
        for _ in range(0, size):
            job = pool.apply_async(createExampleSample, args=(task, sample["parameters"], sample["saddlePoint"], sample["eigens"]), callback=callback)
            jobs.append(job)
        [job.wait() for job in jobs]
    bar.finish()

    return dataset