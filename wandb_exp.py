import copy
import pprint
import time
import wandb
wandb.login()

import numpy as np

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
        'method': 'random',
        'metric': {
            'name': 'accuracy_topk',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5
        },
        'parameters': {
            'weight_init':{
                'values': ['normal', 'neuron_importance', 'zeros']
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
            'value': "madelon"
        },
        'K': {
            'value': 20
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
        'batch_size':{
            'value': 32
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
        }
    })

    pprint.pprint(sweep_config)

    # done to here

    def run_exp(config=None):
        sum_training_time = 0
        with wandb.init(config=config):
            config=wandb.config
            np.random.seed(42)
            x_train, y_train, x_test, y_test = get_data(config.data)
            network = SET_MLP((x_train.shape[1], config.nhidden, y_train.shape[1]),
                              (AlternatedLeftReLU(-config.allrelu_slope), Softmax), 
                              input_pruning=config.input_pruning,
                              importance_pruning=config.importance_pruning,
                              epsilon=config.epsilon,
                              lamda=config.lamda,
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
                batch_size=config.batch_size,
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


    sweep_id = wandb.sweep(sweep_config, project="comparing-init")
    wandb.agent(sweep_id, function=run_exp, count=100)

    wandb.finish()