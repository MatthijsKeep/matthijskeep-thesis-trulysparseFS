import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch WAST")

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs for the feature search experiment.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--epsilon", type=int, default=20, help="epsilon to set the sparsity level"
    )
    parser.add_argument("--nhidden", type=int, default=200)
    parser.add_argument(
        "--K", type=int, default=50, help="20 for madelon, 50 for other datasets"
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=0.9,
        help="coefficient in neuron importance criteria",
    )
    parser.add_argument(
        "--zeta",
        type=float,
        default=0.3,
        help="The fraction of dropped and regrown weights",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0002, help="weight decay"
    )
    parser.add_argument(
        "--allrelu_slope",
        type=float,
        default=0.6,
        help="The slope of the Alternated Left ReLU function",
    )

    parser.add_argument("--eval_epoch", type=int, default=5)
    parser.add_argument(
        "--data",
        type=str,
        default="FashionMnist",
        help="madelon, USPS, coil, mnist, FashionMnist, HAR, Isolet, PCMAC, SMK, GLA",
    )

    parser.add_argument(
        "--update_batch", type=bool, default=False, help="Schedule for topology update"
    )
    parser.add_argument(
        "--input_pruning",
        type=bool,
        default=False,
        help="If true, the input layer will be pruned.",
    )
    parser.add_argument(
        "--importance_pruning",
        type=bool,
        default=False,
        help="If true, the importance pruning will be used.",
    )
    parser.add_argument(
        "--hidden_importance",
        type=bool,
        default=False,
        help="If true, the hidden importance will be used.",
    )

    parser.add_argument(
        "--plotting",
        type=bool,
        default=False,
        help="If true, some plotting will be used.",
    )

    return parser
