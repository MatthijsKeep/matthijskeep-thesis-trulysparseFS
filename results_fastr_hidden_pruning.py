import copy
import pprint
import time
from typing_extensions import TypeAlias
import wandb

wandb.login(key="43d952ea50348fd7b9abbc1ab7d0b787571e8918", timeout=300)

import numpy as np

import sys, os

sys.path.append(os.getcwd())

from argparser import get_parser
from wandbtest import (
    SET_MLP,
    get_data,
    setup_logger,
    print_and_log,
    AlternatedLeftReLU,
    Softmax,
    CrossEntropy,
    select_input_neurons,
    evaluate_fs,
)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print("*******************************************************")
    setup_logger(args)
    print_and_log(args)
    print("*******************************************************")
    runs = args.runs
    sum_training_time = 0
    accuracies = []
    # load data
    no_training_samples = 50000  # max 60000 for Fashion MNIST
    no_testing_samples = 10000  # max 10000 for Fashion MNIST

    # create wandb run

    sweep_config = {
        "method": "grid",
        "metric": {"name": "accuracy_topk", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 50},
        "parameters": {
            "data": {
                "distribution": "categorical",
                "values": [
                    # "synthetic1",
                    # "synthetic2",
                    # "synthetic3",
                    # "synthetic4",
                    # "synthetic5",
                    "mnist",
                    # "madelon",
                    # "smk",
                    # "gla",
                    "usps",
                    "coil",
                    # "isolet",
                    # "har",
                ],
            },
            # 'flex_batch_size':{
            #     'distribution': 'categorical',
            #     'values': [True, False]
            # },
            # 'flex_param':{
            #     'distribution': 'categorical',
            #     'values': [5, 10, 50]
            # },
            # 'input_pruning':{
            #     'distribution': 'categorical',
            #     'values': [True, False]
            # },
            "importance_pruning": {
                "distribution": "categorical",
                "values": [True, False],
            }
            # 'learning_rate':{
            #     'distribution': 'categorical',
            #     'values': [1e-2, 1e-3, 1e-4]
            # },
            # 'lamda':{
            #     'distribution': 'categorical',
            #     'values': [0.95, 0.99]
            # },
            # 'nhidden':{
            #     'distribution': 'categorical',
            #     'values': [100, 200, 500, 1000, 2500]
            # },
            # 'n_redundant':{
            #     'distribution': 'categorical',
            #     'values': [0, 5, 10, 25, 50, 100]
            # },
            # 'weight_decay' :{
            #     'distribution': 'categorical',
            #     'values': [1e-4, 1e-5, 1e-6, 1e-7]
            # },
            # "weight_init": {
            #     "distribution": "categorical",
            #     "values": ["zeros", "normal", "he_uniform"],
            # },
            # 'zeta' : {
            #     'distribution': 'categorical',
            #     'values': [0.2, 0.3, 0.4, 0.5, 0.6]
            # },
        },
    }

    sweep_config["parameters"].update(
        {
            "allrelu_slope": {"value": 0.6},
            "dropout_rate": {"value": 0.3},
            "epochs": {"value": 1000},
            "epsilon": {"value": 20},
            "eval_epoch": {"value": args.eval_epoch},
            "flex_batch_size": {"value": True},
            "flex_param": {"value": 16},
            # "importance_pruning": {"value": True},
            "input_pruning": {"value": False},
            "lamda": {"value": 0.99},
            "learning_rate": {"value": 1e-2},
            "momentum": {"value": args.momentum},
            "nhidden": {"value": 200},
            "n_classes": {"value": 2},
            "n_clusters_per_class": {"value": 16},
            "n_samples": {"value": 500},
            "n_features": {"value": 5000},
            "n_informative": {"value": 20},
            "n_redundant": {"value": 0},
            "n_repeated": {"value": 0},
            "plotting": {"value": True},
            "runs": {"value": args.runs},
            "update_batch": {"value": True},
            "use_neuron_importance": {"value": True},
            "weight_decay": {"value": 1e-5},
            "weight_init": {"value": "zeros"},
            "zero_init_param": {"value": 1e-4},
            "zeta": {"value": 0.3},
        }
    )

    pprint.pprint(sweep_config)

    # done to here

    def run_exp(config=None):
        # print("step 1")
        sum_training_time = 0
        with wandb.init(config=config):
            config = wandb.config
            print(config)
            # print("step 2")
            if config.data == "synthetic":
                data_config = {
                    "n_features": config.n_features,
                    "n_classes": config.n_classes,
                    "n_samples": config.n_samples,
                    "n_informative": config.n_informative,
                    "n_redundant": config.n_redundant,
                    "n_clusters_per_class": config.n_clusters_per_class,
                }
                print(f"data config n_informative: {data_config['n_informative']}")
                x_train, y_train, x_test, y_test, x_val, y_val = get_data(
                    config.data, **data_config
                )
            else:
                x_train, y_train, x_test, y_test, x_val, y_val = get_data(config.data)

            # print("step 3")
            if config.flex_batch_size:
                print(
                    f"The batch size is flexible since flex_batch_size is {config.flex_batch_size}."
                )
                # if the batch size is too large, we ensure that there are at least 8 batches
                batch_size = int(np.ceil(x_train.shape[0] / config.flex_param))
                # round up to the nearest power of 2
                batch_size = 2 ** int(np.ceil(np.log2(batch_size)))
                # ensure that the batch size is never larger than 128
                batch_size = min(batch_size, 128)
                print(batch_size)
            else:
                print(
                    f"The batch size is fixed since flex_batch_size is {config.flex_batch_size}."
                )
                batch_size = 32
            np.random.seed(42)
            # print("step 4")
            network = SET_MLP(
                (x_train.shape[1], config.nhidden, y_train.shape[1]),
                (AlternatedLeftReLU(-config.allrelu_slope), Softmax),
                input_pruning=config.input_pruning,
                importance_pruning=config.importance_pruning,
                epsilon=config.epsilon,
                lamda=config.lamda,
                weight_init=config.weight_init,
                config=config,
            )  # One-layer version
            print(
                f"Data shapes are: {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}, {x_val.shape}, {y_val.shape}"
            )
            metrics = np.zeros((config.runs, config.epochs, 4))
            start_time = time.time()
            # print("step 5")
            network.fit(
                x_train,
                y_train,
                x_test,
                y_test,
                loss=CrossEntropy,
                epochs=config.epochs,
                batch_size=batch_size,
                learning_rate=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                zeta=config.zeta,
                dropoutrate=config.dropout_rate,
                testing=True,
                save_filename=f"results/set_mlp_sequential_{x_train.shape[0]}_training_samples_e{config.epsilon}_rand1",
                monitor=True,
                run=1,
                metrics=metrics,
                eval_epoch=config.eval_epoch,
                config=config,
            )
            print("Training finished")
            print(f"Now selecting {config.K} features")
            selected_features, importances = select_input_neurons(
                copy.deepcopy(network.w[1]), config.K
            )
            time_before_fs = time.time()
            accuracy_topk, pct_correct = evaluate_fs(
                x_train,
                x_val,
                y_train,
                y_val,
                selected_features,
                config.K,
                after_training=True,
            )
            time_after_fs = time.time()
            print("Time for FS after training: ", time_after_fs - time_before_fs)
            wandb.summary["pct_correct"] = pct_correct
            wandb.log({"pct_correct": pct_correct})
            wandb.summary["accuracy_topk"] = accuracy_topk
            wandb.log({"accuracy_topk": accuracy_topk})

            print("Accuracy top k: ", accuracy_topk)
            step_time = time.time() - start_time
            print("\nTotal execution time: ", step_time)
            print("\nTotal training time: ", network.training_time)
            print("\nTotal training time: ", network.training_time)
            print("\nTotal testing time: ", network.testing_time)
            print("\nTotal evolution time: ", network.evolution_time)
            sum_training_time += step_time

    # print("before sweep id")
    sweep_id = wandb.sweep(sweep=sweep_config, project="results-fastr-hidden-pruning")
    # print("before calling agent")
    wandb.agent(sweep_id, function=run_exp)

    wandb.finish()
