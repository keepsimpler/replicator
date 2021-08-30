import torch
import torch.nn as nn

class Trainer(object):

    def __init__(self, model: nn.Module, dataset, criterion, optimizer) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.iterations = 0

    def train_
