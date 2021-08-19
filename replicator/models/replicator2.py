import math
from typing import Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch import einsum
from einops import rearrange, reduce


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
            self.scale = 1 / math.sqrt((self.dim1 + self.dim2)/2)
        self.minimum = - self.scale
        self.maximum = self.scale


@dataclass
class NormalConfig(DistributionConfig):
    mean: float = 0.
    std: float = field(init=False)

    def __post_init__(self):
        if self.scale is None:
            self.scale = 1 / math.sqrt((self.dim1 + self.dim2)/2)
        self.std = self.scale


def uniformal_matrix(row: int, col: int, cfg: UniformConfig):
    "Generate a matrix of size [row, col] uniformly distributed between [minimum, maximum)"
    assert cfg.minimum < cfg.maximum
    return (cfg.minimum - cfg.maximum) * torch.rand(row, col) + cfg.maximum

def normal_matrix(row: int, col: int, cfg: NormalConfig):
    "Generate a matrix of size [row, col] with normal distribution whose mean and std are given"
    return torch.normal(cfg.mean, cfg.std, size=(row, col))

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
    def __init__(self, prob_dim: int, cfg: Union[UniformConfig, NormalConfig], mask: bool=False):
        super().__init__()

        if type(cfg) == UniformConfig:
            weight = uniformal_matrix(prob_dim, prob_dim, cfg)
        elif type(cfg) == NormalConfig:
            weight = normal_matrix(prob_dim, prob_dim, cfg)
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
    def __init__(self, max_sentence_len: int, embedding_size: int, 
                cfg_sentence: Union[UniformConfig, NormalConfig], 
                cfg_embedding: Union[UniformConfig, NormalConfig], 
                mask: bool=False):
        super().__init__()
        self.replicator_sentence = ReplicatorLayer(max_sentence_len, cfg_sentence, mask)
        self.replicator_embedding = ReplicatorLayer(embedding_size, cfg_embedding)

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
    def __init__(self, vocab_size: int, embedding_size: int, cfg: Union[UniformConfig, NormalConfig]):
        super().__init__()

        if type(cfg) == UniformConfig:
            weight = uniformal_matrix(vocab_size, embedding_size, cfg)
        elif type(cfg) == NormalConfig:
            weight = normal_matrix(vocab_size, embedding_size, cfg)
        self.weight = Parameter(weight)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        weight = self.softmax(self.weight)
        # weight[0] = 0
        weight = torch.index_fill(weight, 0, torch.tensor([0]), 0.)
        y = F.embedding(x, weight, padding_idx=0)
        return y


class StochasticProjection(nn.Module):
    """
    将最后一个维度为概率空间分布的tensor，通过stochastic matrix转换为更大空间上的概率分布
    """
    def __init__(self, vocab_size: int, embedding_size: int, cfg: Union[UniformConfig, NormalConfig]):
        super().__init__()

        if type(cfg) == UniformConfig:
            weight = uniformal_matrix(vocab_size, embedding_size, cfg)
        elif type(cfg) == NormalConfig:
            weight = normal_matrix(vocab_size, embedding_size, cfg)
        self.weight = Parameter(weight)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        weight = self.softmax(self.weight)
        y = F.linear(x, weight)
        return y

def log_nll_loss(x_prob, target):
    "输入最后一个维度符合概率空间分布，获得差值"
    x_prob_log = torch.log(x_prob)
    loss = F.nll_loss(x_prob_log, target, ignore_index = 0)
    return loss

class ReplicatorGPT(nn.Module):
    def __init__(self, blocks_num: int, max_sentence_len: int, vocab_size: int, embedding_size: int,
                mask: bool=False):
        super().__init__()

        cfg = NormalConfig(dim1=vocab_size, dim2=embedding_size, scale=1.)
        self.stochastic_embedding = StochasticEmbedding(vocab_size, embedding_size, cfg)

        cfg_sentence = UniformConfig(dim1=max_sentence_len, dim2=max_sentence_len)
        cfg_embedding = UniformConfig(dim1=embedding_size, dim2=embedding_size)
        replicator_blocks = [ReplicatorBlock(max_sentence_len, embedding_size, cfg_sentence, cfg_embedding, mask)
                             for _ in range(blocks_num)]
        self.replicator_blocks = nn.Sequential(*replicator_blocks)

        cfg = NormalConfig(dim1=vocab_size, dim2=embedding_size, scale=1.)
        self.stochastic_projection = StochasticProjection(vocab_size, embedding_size, cfg)

    def forward(self, x, target):
        # (batch_size, max_sentence_len)
        x = self.stochastic_embedding(x)  
        # --> (batch_size, max_sentence_len, embedding_size)
        x = self.replicator_blocks(x)
        # --> (batch_size, max_sentence_len, embedding_size)
        x = self.stochastic_projection(x)
        # --> (batch_size, max_sentence_len, vocab_size)
        tokens_probabilities_exist = torch.sum(x, dim=-1).bool()  # Exclude tokens where all probabilities degrade to 0
        x = x[tokens_probabilities_exist, :]
        target = target[tokens_probabilities_exist]
        loss = log_nll_loss(x, target)
        return loss