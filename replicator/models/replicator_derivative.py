import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torch import einsum

class Replicator2Layer(nn.Module):
    """
    Replicator Layer, inspired by replicator dynamics with equations:
    
    x = x * (A x - x^T A x).

    where the payoff matrix A are determined by ......

    Arguments
    ---------
    max_sentence_len : 
    """
    def __init__(self, max_sentence_len: int):
        super().__init__()

        queries_weight = torch.randn(max_sentence_len, max_sentence_len) / math.sqrt(max_sentence_len)
        keys_weight = torch.randn(max_sentence_len, max_sentence_len) / math.sqrt(max_sentence_len)
        self.queries_weight = Parameter(queries_weight)
        self.keys_weight = Parameter(keys_weight)
        self.max_sentence_len = max_sentence_len

    def forward(self, inputs):
        "`inputs` size (batch_size, max_sentence_len, embedding_size)"

        # repeated `max_sentence_len` times
        queries_weight_repeated = self.queries_weight.repeat(self.max_sentence_len, 1, 1)
        for i in range(self.max_sentence_len):
            queries_weight_repeated[i, :, i+1:] = 0.
            queries_weight_repeated[i, i+1:, :] = 0.

        keys_weight_repeated = self.keys_weight.repeat(self.max_sentence_len, 1, 1)
        for i in range(self.max_sentence_len):
            keys_weight_repeated[i, :, i+1:] = 0.
            keys_weight_repeated[i, i+1:, :] = 0.

        # b --> batch_size, m --> embedding_size, n --> max_sentence_len,  p --> max_sentence_len
        queries = torch.einsum('b n m, ... n p -> b ... m p', inputs, queries_weight_repeated)

        # b --> batch_size, m --> embedding_size, n --> max_sentence_len,  p --> max_sentence_len
        keys = torch.einsum('b n m, ... n p -> b ... p m', inputs, keys_weight_repeated)

        # m --> embedding_size, n --> max_sentence_len, q --> embedding_size
        attentions = torch.einsum('... m n, ... n q -> ... m q', queries, keys)

        # compute Ax, to get fitnesses
        # m --> embedding_size, n --> max_sentence_len, p --> embedding_size,
        fitnesses = torch.einsum('... n m p, b n p -> ... n m', attentions, inputs)

        # compute x * Ax, then summarize through the `embedding_size` dim, in order to get average fitness
        # m --> embedding_size
        avg_fitness = torch.einsum('... m, ... m -> ...', inputs, fitnesses)

        # unsqueeze the `embedding_size` dim
        avg_fitness = avg_fitness.unsqueeze(-1)

        # compute Ax - x * Ax, to get the net fitnesses
        net_fitnesses = fitnesses - avg_fitness

        # compute derivative of inputs
        inputs_derivate = inputs * net_fitnesses

        # Eluer method of ode
        inputs_next = inputs + inputs_derivate

        return inputs_next


class Replicator2GPT(nn.Module):
    """
    Replicator Model similar to GPT model.
    """
    def __init__(self, layers_num: int, max_sentence_len: int, vocab_size: int, embedding_size: int):
        super().__init__()
        # weight的N(0,1)分布是否合适，因为后面又softmax操作 ？
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.softmax = nn.Softmax(dim=-1)
        replicator_layers = [Replicator2Layer(max_sentence_len) for _ in range(layers_num)]
        self.replicator_layers = nn.Sequential(*replicator_layers)
        self.projection = nn.Linear(embedding_size, vocab_size, bias=False)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, inputs, targets, masks):
            inputs_embedding = self.embedding(inputs)
            inputs_embedding[torch.logical_not(masks)] = float('-inf')
            inputs_embedding_softmax = torch.nan_to_num(self.softmax(inputs_embedding))
            outputs = self.replicator_layers(inputs_embedding_softmax)
            logit = self.projection(outputs)
            logit = logit.permute(0,2,1)
            return self.loss(logit, targets)
