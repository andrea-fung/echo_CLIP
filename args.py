import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloader",
    )

    parser.add_argument(
        "--results-db",
        type=str,
        default="/workspace/echo_CLIP/logs",
        help="",
    )

    parser.add_argument(
        "--save",
        type=str,
        default="/workspace/echo_CLIP/logs/finetuned_endtoend",
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )

    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate."
    )

    parser.add_argument(
        "--sampler",
        type=str,
        default='random'
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='train'
    )

    # Hyperparameters for dataset. 
    parser.add_argument(
        "--view",
        type=str,
        default='all'
    )

    parser.add_argument(
        "--flip_rate",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--label_scheme_name",
        type=str,
        default='all'
    )
    # must be compatible with number of unique values in label scheme
    # will be automatic in future update
    # number of AS classes
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4
    )

    #Hyperaparameters for tabular dataset.
    parser.add_argument(
        "--use_tab",
        type=bool,
        default=True
    )

    parser.add_argument(
        "--scale_feats",
        type=bool,
        default=True
    )

    parser.add_argument(
        "--drop_cols",
        type=list,
        default=[]
    )

    parser.add_argument(
        "--categorical_cols",
        type=list,
        default=[]
    )

    # parser.add_argument(
    #     "--data-location",
    #     type=str,
    #     default=os.path.expanduser('~/data'),
    #     help="The root directory for the datasets.",
    # )
    # parser.add_argument(
    #     "--eval-datasets",
    #     default=None,
    #     type=lambda x: x.split(","),
    #     help="Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
    #          " Note that same model used for all datasets, so much have same classnames"
    #          "for zero shot.",
    # )
    # parser.add_argument(
    #     "--train-dataset",
    #     default=None,
    #     help="For fine tuning or linear probe, which dataset to train on",
    # )
    # parser.add_argument(
    #     "--template",
    #     type=str,
    #     default=None,
    #     help="Which prompt template is used. Leave as None for linear probe, etc.",
    # )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--alpha",
        default=[0.5],
        nargs='*',
        type=float,
        help=(
            'Interpolation coefficient for ensembling. '
            'Users should specify N-1 values, where N is the number of '
            'models being ensembled. The specified numbers should sum to '
            'less than 1. Note that the order of these values matter, and '
            'should be the same as the order of the classifiers being ensembled.'
        )
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500, #TODO
    )

    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    # parser.add_argument(
    #     "--fisher",
    #     type=lambda x: x.split(","),
    #     default=None,
    #     help="TODO",
    # )
    # parser.add_argument(
    #     "--fisher_floor",
    #     type=float,
    #     default=1e-8,
    #     help="TODO",
    # )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args