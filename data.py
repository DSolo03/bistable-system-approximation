from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import scipy.integrate as integrate
from numpy.typing import NDArray
from progress.bar import IncrementalBar
from sklearn.preprocessing import normalize
from task import Parameters, Task

@dataclass
class DatasetSample:
    start: NDArray
    parameters: Parameters
    saddle_point: NDArray
    eigens: list
    indicator: int

def check_sample(task: Task, start: NDArray, parameters: Parameters, time: float = 10000, step: float = 0.01, method: str = "Radau") -> int:
    func = lambda _,z: task.system(z, parameters)
    solution = integrate.solve_ivp(func, [0, time], start, t_eval = np.arange(0, time, step), method = method, first_step = step)

    return task.classify_point(solution.y[:,-1])

def create_dataset_sample(task: Task) -> DatasetSample:
    while True: 
        start, parameters = task.sample()

        if not task.multistable_condition(parameters):
            continue   

        saddle_point, eigens = task.get_saddle(parameters)
        if saddle_point is None or eigens is None:
            continue

        indicator = check_sample(task, start, parameters)
        
        if indicator == 0:
            continue

        return DatasetSample(start, parameters, saddle_point, eigens, indicator)

def generate_dataset(task: Task, size: int = 1000, processes: int = 6) -> list[DatasetSample]:
    bar = IncrementalBar('[Dataset] Generating', max = size)
    dataset = []

    def callback(sample):
        dataset.append(sample)
        bar.next()

    create_dataset_sample(task) # Dry run

    print(f"[Dataset] Creating dataset of size {size} samples, on {processes} processes..")  
    with Pool(processes) as pool:
        jobs = []
        for _ in range(0, size):
            job = pool.apply_async(create_dataset_sample, args=(task,), callback=callback)
            jobs.append(job)
        [job.wait() for job in jobs]
    bar.finish()

    return dataset

def create_example_sample(task: Task, sample: DatasetSample) -> DatasetSample:
    start,_ = task.sample()
    indicator = check_sample(task, start, sample.parameters)

    return DatasetSample(start, sample.parameters, sample.saddle_point, sample.eigens, indicator)

def generate_example(task: Task, size: int = 200, processes: int = 6) -> list[DatasetSample]:
    bar = IncrementalBar('[Example] Generating', max = size)
    dataset = []

    def callback(sample):
        dataset.append(sample)
        bar.next()

    sample = create_dataset_sample(task)

    print(f"[Example] Creating dataset of size {size} samples, on {processes} processes..")  
    with Pool(processes) as pool:
        jobs = []
        for _ in range(0, size):
            job = pool.apply_async(create_example_sample, args=(task, sample), callback=callback)
            jobs.append(job)
        [job.wait() for job in jobs]
    bar.finish()

    return dataset