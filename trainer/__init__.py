from .BaseTrainer import BaseTrainer


def get_trainer(trainer):
    return eval(trainer)