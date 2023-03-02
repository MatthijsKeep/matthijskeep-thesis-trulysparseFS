import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch WAST')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--l2', type=float, default=0.00000)

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--eval_epoch', type=int, default=5)
    parser.add_argument('--save-model', type=str, default='./models/model.pt', help='For Saving the current Model')

    parser.add_argument('--nhidden', type=int, default=200)
    parser.add_argument('--data', type=str, default='FashionMnist', help='madelon, USPS, coil, mnist, FashionMnist, HAR, Isolet, PCMAC, SMK, GLA')
    parser.add_argument('--valid_split', type=float, default=0.0)
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--K', type=int, default=50, help='20 for madelon, 50 for other datasets')
    parser.add_argument('--lamda', type=float, default=0.9, help='coefficient in neuron importance criteria')
    parser.add_argument('--density', type=float, default=0.2) 
    parser.add_argument('--alpha', type=float, default=0.3, 
                help='The fraction of dropped and regrown weights')
    parser.add_argument('--allrelu_slope', type=float, default=0.6, help='The slope of the Alternated Left ReLU function')
    parser.add_argument('--rmv_strategy', type=str, default='rmv_IMP',
                    help='rmv weights based on magnitute and connected neuron (IMP) or magnitute only. Options: rmv_IMP, magnitute')
    parser.add_argument('--add_strategy', type=str, default='add_IMP', 
                help='add weights based on neuron importance or random. Options: add_IMP, random')  
    parser.add_argument('--strength', type=str, default='IMP', 
                help='The critreia for selecting the informative features. Options: IMP or weight.') 
    parser.add_argument('--hidden_IMP', type=bool, default=False, help='Always False for WAST. The neurons are equally important')
    parser.add_argument('--update_batch', type=bool, default=True, help='Schedule for topology update')

    parser.add_argument('--model', type=str, default='MLP', help='For the original, WAST implementation use \'WAST\', which will implement the AE model. For the MLP model, use \'MLP\'')
    parser.add_argument('--feature_search_exp', type=bool, default=False, help='If the code is running for the feature search experiment, set this to True. This will run the testing for different values of K.')
    parser.add_argument('--plot_features', type=bool, default=False, help='If true. This will plot the features for different values of K.')
    parser.add_argument('--plot_importances', type=bool, default=False, help='If true. This will plot the neuron importances for different values of K.')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for the feature search experiment.')
    
    return parser