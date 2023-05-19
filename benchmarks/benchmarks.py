from lassonet_fs import lassonet_fs
from multitaskelasticnetcv import multitask_fs
from stg_fs import stg_fs
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import SGDClassifier, MultiTaskElasticNetCV

# import load_in_data from test.py one folder up
from test import load_in_data, svm_test

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def run_feature_selection_baselines(data="madelon", models=None):
    if models is None:
        raise ValueError("No models specified")

    K = 20 if data == "madelon" else 50

    metrics = {model: {"accuracy": []} for model in models}

    for model in models:
        if model == "MultiTaskElasticNetCV":
            print(f"\n Running {model} on {data} dataset \n")
            train_X_new, train_y, test_X_new, test_y = multitask_fs(data, k=K)
            acc = svm_test(train_X_new, train_y, test_X_new, test_y)
            metrics["MultiTaskElasticNetCV"]["accuracy"].append(acc)
        if model == "stochastic_gates":
            print(f"\n Running {model} on {data} dataset \n")
            train_X_new, train_y, test_X_new, test_y = stg_fs(data, k=K)
            acc = svm_test(train_X_new, train_y, test_X_new, test_y)
            metrics["stochastic_gates"]["accuracy"].append(acc)
        if model == "LassoNet":
            print(f"\n Running {model} on {data} dataset \n")
            train_X_new, train_y, test_X_new, test_y = lassonet_fs(data, K)
            acc = svm_test(train_X_new, train_y, test_X_new, test_y)
            metrics["LassoNet"]["accuracy"].append(acc)

    # Save the metrics
    with open(f"./results/{data}_feature_selection_metrics.json", "w") as f:
        json.dump(metrics, f)

    print(f"Metrics saved to ./results/{data}_feature_selection_metrics.json")

    # sort the metrics by accuracy
    metrics = dict(
        sorted(metrics.items(), key=lambda item: item[1]["accuracy"], reverse=True)
    )

    # print the metrics and round to 3 decimal places
    print(f"\n\nMetrics for {data} dataset: \n")
    for model, metric in metrics.items():
        print(f"Model: {model}")
        for metric_name, metric_value in metric.items():
            print(f"{metric_name}: {round(np.mean(metric_value), 3)}")


if __name__ == "__main__":
    run_feature_selection_baselines(
        data="madelon", models=["MultiTaskElasticNetCV", "stochastic_gates", "LassoNet"]
    )
