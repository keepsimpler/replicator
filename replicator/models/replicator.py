import math
from typing import Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch import einsum
from einops import rearrange, reduce

import pytorch_lightning as pl

# import logging
# logging.basicConfig(
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     level=logging.INFO,
#     filename='test.log',
#     filemode='a',
#     datefmt='%Y-%m-%d %H:%M:%S')

@dataclass
class DistributionConfig:
    dim1: int
    dim2: int
    scale: Union[float, None] = None

@dataclass
class UniformConfig(DistributionConfig):
    minimum: float = field(init=False)
    maximum: float = field(init=False)

    def __post_init__(self):
        if self.scale is None:
            #  Xavier initialization
            self.scale = math.sqrt(3.) / math.sqrt((self.dim1 + self.dim2)/2)
        self.minimum = - self.scale
        self.maximum = self.scale


@dataclass
class NormalConfig(DistributionConfig):
    mean: float = 0.
    std: float = field(init=False)

    def __post_init__(self):
        if self.scale is None:
            #  Xavier initialization
            self.scale = 1 / math.sqrt((self.dim1 + self.dim2)/2)
        self.std = self.scale


def uniformal_matrix(cfg: UniformConfig):
    "Generate a matrix of size [row, col] uniformly distributed between [minimum, maximum)"
    assert cfg.minimum < cfg.maximum
    return (cfg.minimum - cfg.maximum) * torch.rand(cfg.dim1, cfg.dim2) + cfg.maximum

def normal_matrix(cfg: NormalConfig):
    "Generate a matrix of size [row, col] with normal distribution whose mean and std are given"
    return torch.normal(cfg.mean, cfg.std, size=(cfg.dim1, cfg.dim2))


def add_up_to_non_negative(tensor: torch.Tensor):
    """
    Eliminate negative values by adding the minimum value up to zero
    through by the last dimension.
    """
    min_values, _ = tensor.min(dim=-1, keepdim=True)
    sub_values = torch.minimum(min_values, torch.zeros_like(tensor))
    return tensor - sub_values

def divide_by_sum(tensor: torch.Tensor):
    """获得类似于softmax的效果。前提是all values are non negative."""
    return tensor / tensor.sum(dim=-1, keepdim=True)

class ReplicatorLayer(nn.Module):
    """
    Replicator Layer, inspired by replicator dynamics with equations:
    
    x = x * (A x - x^T A x).

    Arguments
    ---------
    prob_dim : int, the dimension size of probability space
    cfg : weight initialization configuration
    mask : every probability is only influenced by itself and probabilities before
    
    """
    def __init__(self, cfg: Union[UniformConfig, NormalConfig], mask: bool=False):
        super().__init__()

        if type(cfg) == UniformConfig:
            weight = uniformal_matrix(cfg)
        elif type(cfg) == NormalConfig:
            weight = normal_matrix(cfg)
        self.weight = Parameter(weight)

        self.mask = mask
        if self.mask:
            with torch.no_grad():
                mask_matrix = torch.tril(torch.ones_like(self.weight, dtype=torch.bool), diagonal=0)
                self.register_buffer("mask_matrix", mask_matrix)

    def forward(self, x):
        weight = self.weight
        if self.mask:
            weight = weight.masked_fill(self.mask_matrix, 0)
        # 1. compute Ax
        fitnesses = torch.einsum('m n, ... n -> ... m', weight, x)
        # 2. compute x * Ax, Hadamard product (element-wise product) between x and Ax
        #    then, summed through $n$ dimension
        avg_fitness = torch.einsum('... m, ... m -> ...', x, fitnesses)
        # 3. unsqueeze at the $n$ dimension
        avg_fitness = rearrange(avg_fitness, '... -> ... 1')
        # 3. compute Ax - x * Ax, the net fitnesses
        net_fitnesses = fitnesses - avg_fitness
        # 4. compute  derivative of x
        x_derivative = x * net_fitnesses
        # 5. Eluer method of ode
        x_next = x + x_derivative
        # 6. 
        x_next = F.relu(x_next)
        # x_next = add_up_to_non_negative(x_next)
        return x_next

def swap_stochastic(x):
    "交换最后两个维度，然后将最后一个维度转换为概率空间的分布"
    # 1. swap the last two dimensions
    x = rearrange(x, '... m n -> ... n m')
    # 2. marginal probability of the last dimension
    marginal_prob = reduce(x, '... n m -> ... n', 'sum')
    marginal_prob = rearrange(marginal_prob, '... n -> ... n 1')
    y = x / marginal_prob
    # y[y != y] = 0
    y = torch.nan_to_num(y)  # TODO: may need a better work out
    return y

class ReplicatorBlock(nn.Module):
    def __init__(self, cfg_sentence: Union[UniformConfig, NormalConfig], 
                cfg_embedding: Union[UniformConfig, NormalConfig], 
                mask: bool=False):
        super().__init__()
        self.replicator_sentence = ReplicatorLayer(cfg_sentence, mask)
        self.replicator_embedding = ReplicatorLayer(cfg_embedding)

    def forward(self, x):
        x = self.replicator_embedding(x)
        x = swap_stochastic(x)
        x = self.replicator_sentence(x)
        x = swap_stochastic(x)
        return x

class StochasticEmbedding(nn.Module):
    """
    与nn.Embedding类似，只是weight矩阵通过softmax
    才能作为ReplicatorLayer的输入
    """
    def __init__(self, cfg: Union[UniformConfig, NormalConfig]):
        super().__init__()

        if type(cfg) == UniformConfig:
            weight = uniformal_matrix(cfg)
        elif type(cfg) == NormalConfig:
            weight = normal_matrix(cfg)
        self.weight = Parameter(weight)

        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("padding_idx_tensor", torch.tensor([0]))

    def forward(self, x):
        weight = self.softmax(self.weight)
        # weight[0] = 0
        weight = torch.index_fill(weight, 0, self.padding_idx_tensor, 0.)
        y = F.embedding(x, weight, padding_idx=0)
        return y


class StochasticProjection(nn.Module):
    """
    将最后一个维度为概率空间分布的tensor，通过stochastic matrix转换为更大空间上的概率分布
    """
    def __init__(self, cfg: Union[UniformConfig, NormalConfig]):
        super().__init__()

        if type(cfg) == UniformConfig:
            weight = uniformal_matrix(cfg)
        elif type(cfg) == NormalConfig:
            weight = normal_matrix(cfg)
        self.weight = Parameter(weight)

        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        # weight = self.softmax(self.weight)
        y = F.linear(x, self.weight)
        return y

def log_nll_loss(x_prob, target):
    "输入最后一个维度符合概率空间分布，获得差值"
    x_prob_log = torch.log(x_prob)
    loss = F.nll_loss(x_prob_log, target, ignore_index = 0)
    return loss

class ReplicatorGPT(pl.LightningModule):
    def __init__(self, blocks_num: int, max_sentence_len: int, vocab_size: int, embedding_size: int,
                 lr: float=5e-3, mask: bool=True):
        super().__init__()

        # cfg = NormalConfig(dim1=vocab_size, dim2=embedding_size, scale=1.)
        # self.stochastic_embedding = StochasticEmbedding(cfg)

        cfg_sentence = UniformConfig(dim1=max_sentence_len, dim2=max_sentence_len)
        cfg_embedding = UniformConfig(dim1=embedding_size, dim2=embedding_size)
        replicator_blocks = [ReplicatorBlock(cfg_sentence, cfg_embedding, mask)
                             for _ in range(blocks_num)]
        self.replicator_blocks = nn.Sequential(*replicator_blocks)

        cfg = NormalConfig(dim1=vocab_size, dim2=embedding_size, scale=1.)
        self.stochastic_projection = StochasticProjection(cfg)
        
        # weight的N(0,1)分布是否合适，因为后面又softmax操作 ？
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        # self.embedding.weight = self.stochastic_projection.weight  # tied weight
        self.softmax = nn.Softmax(dim=-1)
        self.lr = lr

    def forward(self, x, masks):
        inputs_embedding = self.embedding(x)
        inputs_embedding[torch.logical_not(masks)] = float('-inf')
        x = torch.nan_to_num(self.softmax(inputs_embedding))
        # x = self.softmax(inputs_embedding)

        # (batch_size, max_sentence_len)
        # x = self.stochastic_embedding(x)  
        # --> (batch_size, max_sentence_len, embedding_size)
        x = self.replicator_blocks(x)
        # --> (batch_size, max_sentence_len, embedding_size)
        x = self.stochastic_projection(x)
        # --> (batch_size, max_sentence_len, vocab_size)
        return x

    def training_step(self, batch, batch_idx):
        x, target, masks = batch
        target = target.clone()
        x = self.forward(x, masks)
        # --> (batch_size, max_sentence_len, vocab_size)
        tokens_probabilities_exist = torch.sum(x, dim=-1).bool()  # Exclude tokens where all probabilities degrade to 0
        x = x[tokens_probabilities_exist, :]
        target = target[tokens_probabilities_exist]
        if not torch.all(tokens_probabilities_exist == masks):
            probabilities_all_zeros = torch.numel(masks[tokens_probabilities_exist != masks])
            probabilities = torch.numel(masks)
            diff_percent = probabilities_all_zeros / probabilities
            # logging.info(f'probabilities\t{probabilities_all_zeros}\t{probabilities}\t{diff_percent}')
            self.log('probabilities_all_zeros', diff_percent)
        loss = F.cross_entropy(x, target, ignore_index=0)
        # loss = log_nll_loss(x, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target, masks = batch
        target = target.clone()
        x = self.forward(x, masks)
        # --> (batch_size, max_sentence_len, vocab_size)
        tokens_probabilities_exist = torch.sum(x, dim=-1).bool()  # Exclude tokens where all probabilities degrade to 0
        x = x[tokens_probabilities_exist, :]
        target = target[tokens_probabilities_exist]
        if not torch.all(tokens_probabilities_exist == masks):
            probabilities_all_zeros = torch.numel(masks[tokens_probabilities_exist != masks])
            probabilities = torch.numel(masks)
            diff_percent = probabilities_all_zeros / probabilities
            # logging.info(f'probabilities\t{probabilities_all_zeros}\t{probabilities}\t{diff_percent}')
            self.log('probabilities_all_zeros', diff_percent)
        loss = F.cross_entropy(x, target, ignore_index=0)
        # loss = log_nll_loss(x, target)
        self.log('valid_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        # scheduler = 
        return optimizer



def mask(inputs, mask_token_id: int, vocab_size: int, p1: float, p2: float, p3: float):
  """
  """
  inputs = inputs.clone()
  inputs_mask1 = torch.rand_like(inputs, dtype=torch.float) < p1
  inputs_mask2 = inputs_mask1 & (torch.rand_like(inputs, dtype=torch.float) < p2 / p1)
  inputs_mask2[inputs <= 1] = False  # Do not mask special tokens
  inputs[inputs_mask2] = mask_token_id  # 
  inputs_mask3 = inputs_mask2 & (torch.rand_like(inputs, dtype=torch.float) < p3 / p2)
  inputs[inputs_mask3] = torch.randint(2, vocab_size - 1, (inputs_mask3.sum().item(),), device='cuda:0')

  # loss weights
  loss_weights = torch.zeros_like(inputs)
  loss_weights[inputs_mask1] = 1

  return inputs, loss_weights


class ReplicatorBERT(pl.LightningModule):
    def __init__(self, blocks_num: int, max_sentence_len: int, vocab_size: int, embedding_size: int, 
                 p1: float, p2: float, p3: float,
                 lr: float=5e-3, mask: bool=False):
        super().__init__()

        # cfg = NormalConfig(dim1=vocab_size, dim2=embedding_size, scale=1.)
        # self.stochastic_embedding = StochasticEmbedding(cfg)

        cfg_sentence = UniformConfig(dim1=max_sentence_len, dim2=max_sentence_len)
        cfg_embedding = UniformConfig(dim1=embedding_size, dim2=embedding_size)
        replicator_blocks = [ReplicatorBlock(cfg_sentence, cfg_embedding, mask=False)
                             for _ in range(blocks_num)]
        self.replicator_blocks = nn.Sequential(*replicator_blocks)

        cfg = NormalConfig(dim1=vocab_size, dim2=embedding_size, scale=1.)
        self.stochastic_projection = StochasticProjection(cfg)
        
        # weight的N(0,1)分布是否合适，因为后面又softmax操作 ？
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        # self.embedding.weight = self.stochastic_projection.weight  # tied weight
        self.softmax = nn.Softmax(dim=-1)
        self.lr = lr
        self.vocab_size = vocab_size
        self.p1, self.p2, self.p3 = p1, p2, p3

    def forward(self, x, masks):
        inputs_embedding = self.embedding(x)
        inputs_embedding[torch.logical_not(masks)] = float('-inf')
        x = torch.nan_to_num(self.softmax(inputs_embedding))
        # x = self.softmax(inputs_embedding)

        # (batch_size, max_sentence_len)
        # x = self.stochastic_embedding(x)  
        # --> (batch_size, max_sentence_len, embedding_size)
        x = self.replicator_blocks(x)
        # --> (batch_size, max_sentence_len, embedding_size)
        x = self.stochastic_projection(x)
        # --> (batch_size, max_sentence_len, vocab_size)
        return x

    def training_step(self, batch, batch_idx):
        x, target, masks = batch
        x, loss_weights = mask(x, 0, self.vocab_size, p1=self.p1, p2=self.p2, p3=self.p3)
        x = self.forward(x, masks)
        # --> (batch_size, max_sentence_len, vocab_size)
        tokens_probabilities_exist = torch.sum(x, dim=-1).bool()  # Exclude tokens where all probabilities degrade to 0
        x = x[tokens_probabilities_exist, :]
        target = target[tokens_probabilities_exist]
        loss_weights = loss_weights[tokens_probabilities_exist]
        if not torch.all(tokens_probabilities_exist == masks):
            probabilities_all_zeros = torch.numel(masks[tokens_probabilities_exist != masks])
            probabilities = torch.numel(masks)
            diff_percent = probabilities_all_zeros / probabilities
            # logging.info(f'probabilities\t{probabilities_all_zeros}\t{probabilities}\t{diff_percent}')
            self.log('probabilities_all_zeros', diff_percent)
        loss = F.cross_entropy(x, target, ignore_index=0, reduction='none') * loss_weights
        loss = loss.sum() / loss_weights.sum()
        # loss = log_nll_loss(x, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target, masks = batch
        x, loss_weights = mask(x, 0, self.vocab_size, p1=self.p1, p2=self.p2, p3=self.p3)
        x = self.forward(x, masks)
        # --> (batch_size, max_sentence_len, vocab_size)
        tokens_probabilities_exist = torch.sum(x, dim=-1).bool()  # Exclude tokens where all probabilities degrade to 0
        x = x[tokens_probabilities_exist, :]
        target = target[tokens_probabilities_exist]
        loss_weights = loss_weights[tokens_probabilities_exist]
        if not torch.all(tokens_probabilities_exist == masks):
            probabilities_all_zeros = torch.numel(masks[tokens_probabilities_exist != masks])
            probabilities = torch.numel(masks)
            diff_percent = probabilities_all_zeros / probabilities
            # logging.info(f'probabilities\t{probabilities_all_zeros}\t{probabilities}\t{diff_percent}')
            self.log('probabilities_all_zeros', diff_percent)
        loss = F.cross_entropy(x, target, ignore_index=0, reduction='none') * loss_weights
        loss = loss.sum() / loss_weights.sum()
        # loss = log_nll_loss(x, target)
        self.log('valid_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        # scheduler = 
        return optimizer
