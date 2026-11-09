#!/usr/bin/env python3

import argparse

from sgimda.config import SGIMDAConfig
from sgimda.training import train_fold

DATASETS = ("HMDDv4.0", "miRBase")
PROTOCOLS = ("random", "cold_disease", "cold_mirna")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", nargs="+", choices=DATASETS, required=True)
    parser.add_argument("--protocol", nargs="+", choices=PROTOCOLS, required=True)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 6),
                        default=list(range(1, 6)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers:
        parser.error("worker must satisfy 0 <= worker < workers")
    return args

def main():
    args = parse_args()
    config = (
        SGIMDAConfig.from_yaml(args.config)
        if args.config
        else SGIMDAConfig().validate()
    )
    tasks = [
        (dataset, protocol, fold)
        for dataset in dict.fromkeys(args.dataset)
        for fold in sorted(set(args.folds))
        for protocol in dict.fromkeys(args.protocol)
    ]
    for index, (dataset, protocol, fold) in enumerate(tasks):
        if index % args.workers == args.worker:
            train_fold(
                args.data_root,
                args.splits_root,
                args.output,
                dataset, protocol, fold, args.device, config,
            )

if __name__ == "__main__":
    main()
