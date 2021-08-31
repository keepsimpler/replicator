import math

import torch
import torch.nn as nn
from accelerate import Accelerator

from replicator.utils import Accumulator

class Trainer(object):

    def __init__(self, model: nn.Module, dataset, optimizer) -> None:
        super().__init__()
        # self.model = model
        # self.dataset = dataset
        # self.optimizer = optimizer

        self.accelerator = Accelerator()        

        self.model, self.dataset, self.optimizer = self.accelerator.prepare(model, dataset, optimizer)

        self.iterations = 0

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            self.model.train()
            epoch_metric = Accumulator(2)
            for inputs, targets, masks in self.dataset:
                inputs = inputs.to(self.accelerator.device)
                targets = targets.to(self.accelerator.device)
                masks = masks.to(self.accelerator.device)
                self.optimizer.zero_grad()
                loss = self.model(inputs, targets, masks)
                # with torch.autograd.set_detect_anomaly(True):
                # loss.backward()
                self.accelerator.backward(loss)
                self.optimizer.step()
                with torch.no_grad():
                    epoch_metric.add(loss, 1)
            loss = epoch_metric[0] / epoch_metric[1]
            perplexity = math.exp(epoch_metric[0] / epoch_metric[1])
            print(f'epoch {epoch + 1}, perplexity {float(perplexity):.6f}, loss {float(loss):.6f}')
            # eval_perplexity, eval_loss = evaluate_perplexity(self.model, validate_dataloader)
            # print(f'epoch {epoch + 1}, eval_perplexity {float(eval_perplexity):.6f}, eval_loss {float(eval_loss):.6f}')

