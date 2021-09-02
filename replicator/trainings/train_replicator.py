import math

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    filename='test.log',
    filemode='a',
    datefmt='%Y-%m-%d %H:%M:%S')

import torch
import torch.nn as nn
from accelerate import Accelerator

from replicator.utils import Accumulator, net_statistics

class Trainer(object):

    def __init__(self, model: nn.Module, dataset, optimizer) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer

        self.iterations = 0

    def train(self, num_epochs):
        accelerator = Accelerator()        
        logging.info('start')
        model, dataset, optimizer = accelerator.prepare(self.model, self.dataset, self.optimizer)
        model.train()
        for epoch in range(num_epochs):
            model.train()
            epoch_metric = Accumulator(2)
            for inputs, targets, masks in dataset:
                inputs = inputs.to(accelerator.device)
                targets = targets.to(accelerator.device)
                masks = masks.to(accelerator.device)
                loss = model(inputs, targets, masks)
                optimizer.zero_grad()
                # with torch.autograd.set_detect_anomaly(True):
                # loss.backward()
                accelerator.backward(loss)
                stats = net_statistics(model)
                for s in stats:
                    logging.info(f'stats:\t{s["name"]}\t{s["mean"]}\t{s["std"]}\t{s["grad.abs.mean"]}\t{s["grad.std"]}')
                optimizer.step()
                with torch.no_grad():
                    epoch_metric.add(loss, 1)
            loss = epoch_metric[0] / epoch_metric[1]
            perplexity = math.exp(epoch_metric[0] / epoch_metric[1])
            print(f'epoch {epoch + 1}, perplexity {float(perplexity):.6f}, loss {float(loss):.6f}')
            # eval_perplexity, eval_loss = evaluate_perplexity(self.model, validate_dataloader)
            # print(f'epoch {epoch + 1}, eval_perplexity {float(eval_perplexity):.6f}, eval_loss {float(eval_loss):.6f}')

