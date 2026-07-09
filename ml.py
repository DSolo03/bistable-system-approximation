from typing import cast

import numpy as np
import pandas as pd
from data import DatasetSample
from imblearn.under_sampling import RandomUnderSampler
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from task import Task

def prepare_features(task: Task, dataset: list[DatasetSample]) -> tuple[pd.DataFrame, pd.Series]:
    data = []
    indicator = []
    for row in dataset:
        if row.indicator in [-1,1]:
            terms = task.expansion(row.start, row.parameters, row.saddle_point, row.eigens)
            data = data + terms
            indicator.extend([row.indicator]*len(terms))
    return pd.DataFrame(data), pd.Series(indicator)

def train(task: Task, dataset: list[DatasetSample], balance: bool = True) -> tuple[NDArray, dict]:
    statistics={}

    print("[MachineLearning] Splitting dataset by 20%")
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    X_train, y_train = prepare_features(task, train)
    X_test, y_test = prepare_features(task, test)
    
    X_train.to_csv("data\\dataset.csv")
    
    keep_train = X_train.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    print(f"[MachineLearning] Removed {len(X_train) - keep_train.sum()} rows with inf/nan from train set")
    X_train = X_train[keep_train]
    y_train = y_train[keep_train]

    keep_test = X_test.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    print(f"[MachineLearning] Removed {len(X_test) - keep_test.sum()} rows with inf/nan from test set")
    X_test = X_test[keep_test]
    y_test = y_test[keep_test]

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"[MachineLearning] Scaling training values by coef = {scaler.scale_}")
    firstClass, secondClass = y_train.value_counts()
    
    print(f"[MachineLearning] Classes before balancing: {firstClass}/{secondClass}")
    if balance:
        print("[MachineLearning] Balancing classses")
        firstClass, secondClass = y_train.value_counts()
        rus = RandomUnderSampler(sampling_strategy='auto')
        X_train, y_train = cast(
            tuple[pd.DataFrame, pd.Series], 
            rus.fit_resample(X_train, y_train)
        )
        first_class, second_class = y_train.value_counts()
        print(f"[MachineLearning] Classes after balancing: {first_class}/{second_class}")

    model = LinearSVC(penalty='l2', dual=False,fit_intercept=False, C=0.1, max_iter=300000)
    model.fit(X_train, y_train)

    lambdas = model.coef_[0]

    predictions = model.predict(X_test)

    predictions_proba = model.decision_function(X_test)
    probabilities = 1 / (1 + np.exp(-predictions_proba))

    lambdas = lambdas / scaler.scale_
    divider = lambdas[0]
    for i in range(len(lambdas)):
        lambdas[i] = lambdas[i] / divider

    loss = log_loss(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    statistics.update({"size":len(dataset)})
    statistics.update({"loss":loss})
    statistics.update({"accuracy":accuracy})
    statistics.update({"recall":recall})
    statistics.update({"precision":precision})
    statistics.update({"f1":f1})

    return lambdas, statistics