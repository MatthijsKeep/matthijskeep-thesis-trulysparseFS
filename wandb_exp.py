import copy
import pprint
import time
from typing_extensions import TypeAlias
import wandb

wandb.login(key="91381d9442817977bd94e5b7ebe345cbbe49fd6f", timeout=300)

import numpy as np

import sys, os

sys.path.append(os.getcwd())

from argparser import get_parser
from wandbtest import SET_MLP, get_data, setup_logger, print_and_log, AlternatedLeftReLU, Softmax, CrossEntropy, select_input_neurons, evaluate_fs

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
        'method': 'grid',
        'metric': {
            'name': 'accuracy_topk',
            'goal': 'maximize'
        },
        'parameters': {
            'n_features': {
                'values': [500, 5000, 25000]
            },
            'n_classes': {
                'values': [2, 5, 20]
            },
            'n_samples': {
                'values': [50, 500, 2500]
            },
            'n_clusters_per_class': {
                'values': [1, 4, 16]
            },
            'flex_batch_size': {
                'values': [True, False]
            },
            'flex_param':{
                'values': [8, 16]
            }
        }
    }

    sweep_config["parameters"].update({
        'nhidden': {
            'value': 200},
        'weight_decay': {
            'value': args.weight_decay
        },
        'momentum': {
            'value': args.momentum
        },
        'allrelu_slope': {
            'value': args.allrelu_slope
        },
        'data':{
            'value': "synthetic"
        },
        'K': {
            'value': 50
        },
        'runs': {
            'value': args.runs
        },
        'eval_epoch': {
            'value': args.eval_epoch
        },
        'input_pruning':{
            'value': True
        },
        'update_batch':{
            'value': True
        },
        'learning_rate': {
            'value': 1e-3
        },
        'epochs':{
            'value': 100
        },
        'importance_pruning':{
            'value': True
        },
        'epsilon':{
            'value': 20
        },
        'lamda':{
            'value': 0.9
        },
        'zeta':{
            'value': 0.4
        },
        'dropout_rate':{
            'value': 0.3
        },
        'plotting': {
            'value': False
        },
        'zero_init_param':{
            'value': 1e-4
        },
        'weight_init':{
            'value': 'zeros'
        },
        'n_informative':{
            'value': 50
        }
    })

    pprint.pprint(sweep_config)

    # done to here

    def run_exp(config=None):
        sum_training_time = 0
        with wandb.init(config=config):
            config=wandb.config
            data_config = {
                "n_features": config.n_features,
                "n_classes": config.n_classes,
                "n_samples": config.n_samples,
                "n_informative": config.n_informative,
                "n_redundant": config.n_redundant,
                "n_clusters_per_class": config.n_clusters_per_class,
            }
            x_train, y_train, x_test, y_test = get_data(config.data, **data_config)
            if config.flex_batch_size:
                print(f"The batch size is flexible since flex_batch_size is {config.flex_batch_size}.")
                # if the batch size is too large, we ensure that there are at least 8 batches
                batch_size = int(np.ceil(x_train.shape[0]/config.flex_param))
                # round up to the nearest power of 2
                batch_size = 2**int(np.ceil(np.log2(batch_size)))
                print(batch_size)
            else:
                print(f"The batch size is fixed since flex_batch_size is {config.flex_batch_size}.")
                batch_size = 32
            np.random.seed(42)

            network = SET_MLP((x_train.shape[1], config.nhidden, y_train.shape[1]),
                              (AlternatedLeftReLU(-config.allrelu_slope), Softmax), 
                              input_pruning=config.input_pruning,
                              importance_pruning=config.importance_pruning,
                              epsilon=config.epsilon,
                              lamda=config.lamda,
                              weight_init=config.weight_init,
                              config=config) # One-layer version   
            print(f"Data shapes are: {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}")
            metrics = np.zeros((config.runs, config.epochs, 4))
            start_time = time.time()

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
            selected_features, importances = select_input_neurons(copy.deepcopy(network.w[1]), config.K)
            accuracy_topk = evaluate_fs(x_train, x_test, y_train, y_test, selected_features)
            wandb.summary['accuracy_topk'] = accuracy_topk
            wandb.log({'accuracy_topk': accuracy_topk})
            print("Accuracy top k: ", accuracy_topk)
            step_time = time.time() - start_time
            print("\nTotal execution time: ", step_time)
            print("\nTotal training time: ", network.training_time)
            print("\nTotal training time: ", network.training_time)
            print("\nTotal testing time: ", network.testing_time)
            print("\nTotal evolution time: ", network.evolution_time)
            sum_training_time += step_time 


    sweep_id = wandb.sweep(sweep_config, project="flex-batch-size")
    wandb.agent(sweep_id, function=run_exp)

    wandb.finish()