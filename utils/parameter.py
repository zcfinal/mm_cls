import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--trainer", type=str, default='BaseTrainer')
    parser.add_argument("--dataset", type=str, default='HateMM')
    parser.add_argument("--dataset_version", type=str, default='all')
    parser.add_argument("--model", type=str, default='clip')
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=200)

    args = parser.parse_args()
    return args