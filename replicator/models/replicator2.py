from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import pytorch_lightning as pl


@dataclass
class ReplicatorConfig:
    blocks_num: int
    seq_len: int
    embedding_size: int

    # data
    vocab_size: int
    padding_idx: int
    is_prefix: bool
    num_special_tokens: int
    mask_token_id: int

    # model
    mask: bool
    predicted_num: int
    p1: float
    p2: float
    p3: float

    lr: float


ReplicatorConfigBERT = partial(
    ReplicatorConfig, mask=False, predicted_num=0, p1=0.15, p2=0.9, p3=1/9)
ReplicatorConfigGPT = partial(
    ReplicatorConfig, mask=True, predicted_num=1, p1=0, p2=0, p3=0)
ReplicatorConfigOpenwebtext = partial(ReplicatorConfig, vocab_size=50260,
                                      padding_idx=50257, is_prefix=False, num_special_tokens=3, mask_token_id=50259)
ReplicatorConfigWikitext2 = partial(ReplicatorConfig, vocab_size=28784,
                                    padding_idx=0, is_prefix=True, num_special_tokens=3, mask_token_id=2)

class ReplicatorDerivLayer(nn.Module):
    def __init__(self, seq_len: int, embedding_size: int, mask: bool = False):
        super(ReplicatorDerivLayer, self).__init__()
        weight1 = torch.Tensor(embedding_size, embedding_size)
        nn.init.xavier_uniform_(weight1)
        self.weight1 = Parameter(weight1)
        weight2 = torch.Tensor(embedding_size, embedding_size)
        nn.init.xavier_uniform_(weight2)
        self.weight2 = Parameter(weight2)
        self.mask = mask
        if mask:
            mask_matrix = torch.triu(torch.ones(
                seq_len, seq_len, dtype=torch.bool
            ), diagonal=1)
            self.register_buffer("mask_matrix", mask_matrix)

    def forward(self, x):
        # (batch_size, embedding_size, seq_len)
        fitnesses1 = torch.einsum('b e s, e f -> b f s', x, self.weight1)
        fitnesses2 = torch.einsum('b f s, e f -> b e s', x, self.weight2)
        weight = torch.einsum('b e s, b e t -> b s t', fitnesses1, fitnesses2)
        if self.mask:
            weight = weight.masked_fill(self.mask_matrix, 0)
        # --> (batch_size, seq_len, seq_len)
        # 1. compute Wx, matrix-vector product between W and x
        fitnesses = torch.einsum('b s t, b e t -> b e s', weight, x)
        # 2. compute x^T Wx, dot product between x and Ax
        avg_fitness = torch.einsum('b e s, b e s -> b e', x, fitnesses)
        # 3. unsqueeze at the last dimension,
        # for broadcasting in the following substract operator
        avg_fitness = avg_fitness.unsqueeze(dim=-1)
        # 4. compute Wx - x^T Wx, get the net fitnesses
        net_fitnesses = fitnesses - avg_fitness
        # 5. compute  derivative of x
        x_derivative = x * net_fitnesses
        # 6. Eluer method of ode
        x_next = x + x_derivative
        # 7. remove all negative values
        x_next = F.relu(x_next)
        return x_next

class ReplicatorSeqLayer(nn.Module):
    def __init__(self, seq_len: int, embedding_size: int, mask: bool = False):
        super(ReplicatorSeqLayer, self).__init__()
        embedding_weight = torch.Tensor(embedding_size, embedding_size)
        nn.init.xavier_uniform_(embedding_weight)
        self.embedding_weight = Parameter(embedding_weight)
        seq_weight = torch.Tensor(seq_len, seq_len)
        nn.init.xavier_uniform_(seq_weight)
        self.seq_weight = Parameter(seq_weight)
        self.mask = mask
        if mask:
            mask_matrix = torch.triu(torch.ones(
                seq_len, seq_len, dtype=torch.bool
            ), diagonal=1)
            self.register_buffer("mask_matrix", mask_matrix)

    def forward(self, x):
        # (batch_size, embedding_size, seq_len)
        x_embedding_weight = torch.einsum('b e s, f e -> b f s', x, self.embedding_weight)
        if self.mask:
            seq_weight = self.seq_weight.masked_fill(self.mask_matrix, 0)
        else:
            seq_weight = self.seq_weight
        # 1. compute Wx, matrix-vector product between W and x
        fitnesses = torch.einsum('s t, b e t -> b e s', seq_weight, x_embedding_weight)
        # 2. compute x^T Wx, dot product between x and Ax
        avg_fitness = torch.einsum('b e s, b e s -> b e', x, fitnesses)
        # 3. unsqueeze at the last dimension,
        # for broadcasting in the following substract operator
        avg_fitness = avg_fitness.unsqueeze(dim=-1)
        # 4. compute Wx - x^T Wx, get the net fitnesses
        net_fitnesses = fitnesses - avg_fitness
        # 5. compute  derivative of x
        x_derivative = x * net_fitnesses
        # 6. Eluer method of ode
        x_next = x + x_derivative
        # 7. remove all negative values
        x_next = F.relu(x_next)
        return x_next

class ReplicatorEmbeddingLayer(nn.Module):
    def __init__(self, seq_len: int, embedding_size: int, mask: bool = False):
        super(ReplicatorEmbeddingLayer, self).__init__()
        embedding_weight = torch.Tensor(embedding_size, embedding_size)
        nn.init.xavier_uniform_(embedding_weight)
        self.embedding_weight = Parameter(embedding_weight)
        seq_weight = torch.Tensor(seq_len, seq_len)
        nn.init.xavier_uniform_(seq_weight)
        self.seq_weight = Parameter(seq_weight)
        self.mask = mask
        if mask:
            mask_matrix = torch.triu(torch.ones(
                seq_len, seq_len, dtype=torch.bool
            ), diagonal=1)
            self.register_buffer("mask_matrix", mask_matrix)

    def forward(self, x):
        # (batch_size, seq_len, embedding_size)
        if self.mask:
            seq_weight = self.seq_weight.masked_fill(self.mask_matrix, 0)
        else:
            seq_weight = self.seq_weight
        x_seq_weight = torch.einsum('b s e, t s -> b t e', x, seq_weight)
        # 1. compute Wx, matrix-vector product between W and x
        fitnesses = torch.einsum('e f, b t f -> b t e', self.embedding_weight, x_seq_weight)
        # 2. compute x^T Wx, dot product between x and Ax
        avg_fitness = torch.einsum('b e s, b e s -> b e', x, fitnesses)
        # 3. unsqueeze at the last dimension,
        # for broadcasting in the following substract operator
        avg_fitness = avg_fitness.unsqueeze(dim=-1)
        # 4. compute Wx - x^T Wx, get the net fitnesses
        net_fitnesses = fitnesses - avg_fitness
        # 5. compute  derivative of x
        x_derivative = x * net_fitnesses
        # 6. Eluer method of ode
        x_next = x + x_derivative
        # 7. remove all negative values
        x_next = F.relu(x_next)
        return x_next


class ReplicatorLayer(nn.Module):
    def __init__(self, prob_space_size: int, mask: bool = False):
        super(ReplicatorLayer, self).__init__()
        weight = torch.Tensor(prob_space_size, prob_space_size)
        nn.init.xavier_uniform_(weight)
        self.mask = mask
        if mask:
            mask_matrix = torch.triu(torch.ones(
                prob_space_size, prob_space_size, dtype=torch.bool), diagonal=1)
            self.register_buffer("mask_matrix", mask_matrix)
        self.weight = Parameter(weight)

    def forward(self, x):
        if self.mask:
            weight = self.weight.masked_fill(self.mask_matrix, 0)
        else:
            weight = self.weight
        # 1. compute Wx, matrix-vector product between W and x
        fitnesses = torch.einsum('m n, i j n -> i j m', weight, x)
        # 2. compute x^T Wx, dot product between x and Ax
        avg_fitness = torch.einsum('i j m, i j m -> i j', x, fitnesses)
        # 3. unsqueeze at the last dimension,
        # for broadcasting in the following substract operator
        avg_fitness = avg_fitness.unsqueeze(dim=-1)
        # 4. compute Wx - x^T Wx, get the net fitnesses
        net_fitnesses = fitnesses - avg_fitness
        # 5. compute  derivative of x
        x_derivative = x * net_fitnesses
        # 6. Eluer method of ode
        x_next = x + x_derivative
        # 7. remove all negative values
        x_next = F.relu(x_next)
        return x_next


class SwapProbSpaces(nn.Module):
    def __init__(self):
        super(SwapProbSpaces, self).__init__()

    def forward(self, x):
        # 1. swap the last two dimensions
        x = x.transpose(-1, -2).contiguous()
        # 2. marginal probability of the last dimension
        marginal_prob = x.sum(dim=-1, keepdim=True)
        # 3. divide by marginal probability to get the new probability dimension
        x = torch.divide(x, marginal_prob)
        # 4. avoid divided by zero
        x = torch.nan_to_num(x)
        return x


class ReplicatorBlock(nn.Module):
    def __init__(self, seq_len: int, embedding_size: int, mask: bool = False):
        super(ReplicatorBlock, self).__init__()
        self.replicator_embedding = ReplicatorLayer(
            prob_space_size=embedding_size)
        self.replicator_seq = ReplicatorLayer(
            prob_space_size=seq_len, mask=mask)
        self.swap_prob_spaces = SwapProbSpaces()

    def forward(self, x):
        x = self.replicator_embedding(x)
        x = self.swap_prob_spaces(x)
        x = self.replicator_seq(x)
        x = self.swap_prob_spaces(x)
        return x


class Projection(nn.Module):
    def __init__(self, embedding_size: int, vocab_size: int):
        super(Projection, self).__init__()
        weight = torch.Tensor(vocab_size, embedding_size)
        nn.init.uniform_(weight)
        self.weight = Parameter(weight)

    def forward(self, x):
        # 1. compute Wx, matrix-vector product between W and x
        x = torch.einsum('m n, i j n -> i j m', self.weight, x)
        return x


class ReplicatorEmbeddingNetwork(nn.Module):
    def __init__(self, blocks_num: int, seq_len: int, embedding_size: int, vocab_size: int,
                 padding_idx: int, mask:bool):
        super(ReplicatorEmbeddingNetwork, self).__init__()

        replicator_blocks = [ReplicatorEmbeddingLayer(
            seq_len=seq_len, embedding_size=embedding_size, mask=mask) for _ in range(blocks_num)]
        self.replicator_blocks = nn.Sequential(*replicator_blocks)

        self.projection = Projection(embedding_size, vocab_size)
        self.embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx)
        self.softmax = nn.Softmax(dim=-1)

        # self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        x = self.embedding(x)
        x_prob = self.softmax(x)
        y_prob = self.replicator_blocks(x_prob)
        # kl_div_loss = self.kl_div(x_prob.log(), y_prob)
        # self.log('kl_div_loss', kl_div_loss)
        y = self.projection(y_prob)
        return y


class ReplicatorNetwork(nn.Module):
    def __init__(self, blocks_num: int, seq_len: int, embedding_size: int, vocab_size: int,
                 padding_idx: int, mask:bool):
        super(ReplicatorNetwork, self).__init__()

        replicator_blocks = [ReplicatorBlock(
            seq_len, embedding_size, mask=mask) for _ in range(blocks_num)]
        self.replicator_blocks = nn.Sequential(*replicator_blocks)

        self.projection = Projection(embedding_size, vocab_size)
        self.embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx)
        self.softmax = nn.Softmax(dim=-1)

        # self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        x = self.embedding(x)
        x_prob = self.softmax(x)
        y_prob = self.replicator_blocks(x_prob)
        # kl_div_loss = self.kl_div(x_prob.log(), y_prob)
        # self.log('kl_div_loss', kl_div_loss)
        y = self.projection(y_prob)
        return y


def mlm_prepare_data(inputs, vocab_size: int, is_prefix: bool, num_special_tokens: int,
                     mask_token_id: int, p1: float, p2: float, p3: float):
    """
    Prepare data for MLM (mask language model).
    Inputs have shape (batch_size, seq_len)
    Arguments:
    ----------
    batch_size and seq_len: shape of inputs
    vocab_size: 
        for GPT2Tokenizer: default is 50260(50256 tokens + 1 <endoftext> token + 1 <pad> token
                            + 1 <unk> token + 1 <mask> token)
        for TorchText tokenizer: default is (1 <pad> token + 1 <unk> token + 1 <mask> token + x tokens)
    is_prefix: if the special tokens procede or follow the normal tokens? 
    num_special_tokens: number of special tokens
    mask_token_id: the <mask> token id
    p1: probability of tokens that will be selected at random as masked tokens
    p2: probability of replaced by a special <mask> token or a random token for a masked token
    p3: probability of replaced by a random token for the token replaced by a spceical <mask> token or random token
    """
    # inputs = inputs.clone()
    inputs_mask1 = torch.rand_like(inputs, dtype=torch.float) < p1
    inputs_mask2 = inputs_mask1 & (
        torch.rand_like(inputs, dtype=torch.float) < p2)
    if is_prefix:  # if the special tokens precede the normal tokens
        # Do not mask special tokens
        inputs_mask2[inputs < num_special_tokens] = False
    else:  # if the special tokens follow the normal tokens
        # Do not mask special tokens
        inputs_mask2[inputs >= vocab_size - num_special_tokens] = False
    inputs[inputs_mask2] = mask_token_id  #
    inputs_mask3 = inputs_mask2 & (
        torch.rand_like(inputs, dtype=torch.float) < p3)
    if is_prefix:
        inputs[inputs_mask3] = torch.randint(
            num_special_tokens, vocab_size, (inputs_mask3.sum().item(),), device=inputs.device)
    else:
        inputs[inputs_mask3] = torch.randint(
            0, vocab_size - num_special_tokens, (inputs_mask3.sum().item(),), device=inputs.device)

    # loss weights
    loss_weights = torch.zeros_like(inputs)
    loss_weights[inputs_mask1] = 1
    # loss_weights[torch.logical_not(inputs_mask1)] = 1

    return inputs, loss_weights


class ReplicatorEmbedding(pl.LightningModule):
    def __init__(self, conf: ReplicatorConfig):
        super(ReplicatorEmbedding, self).__init__()
        self.replicator_network = ReplicatorEmbeddingNetwork(blocks_num=conf.blocks_num, seq_len=conf.seq_len,
                                                    embedding_size=conf.embedding_size, vocab_size=conf.vocab_size,
                                                    padding_idx=conf.padding_idx, mask=conf.mask)
        self.conf = conf

    def forward(self, x):
        return self.replicator_network(x)

    def one_step(self, batch, batch_idx):
        inputs = batch[:, :self.conf.seq_len]
        targets = batch[:, self.conf.predicted_num:self.conf.seq_len +
                        self.conf.predicted_num].clone().long()
        if self.conf.p1 > 0:
            inputs, loss_weight = mlm_prepare_data(inputs, vocab_size=self.conf.vocab_size, is_prefix=self.conf.is_prefix,
                                                   num_special_tokens=self.conf.num_special_tokens,
                                                   mask_token_id=self.conf.mask_token_id, p1=self.conf.p1,
                                                   p2=self.conf.p2, p3=self.conf.p3)
        outputs = self.forward(inputs)

        # --> (batch_size, max_sentence_len, vocab_size)
        # Exclude tokens where all probabilities degrade to 0
        tokens_probabilities_exist = torch.sum(outputs, dim=-1).bool()
        outputs = outputs[tokens_probabilities_exist, :]
        targets = targets[tokens_probabilities_exist]
        if self.conf.p1 > 0:
            loss_weight = loss_weight[tokens_probabilities_exist]
            loss = F.cross_entropy(
                outputs, targets, reduction='none') * loss_weight
            loss = loss.sum() / loss_weight.sum()
        else:
            loss = F.cross_entropy(outputs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.one_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.one_step(batch, batch_idx)
        self.log('valid_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        inputs = batch[:, :self.conf.seq_len]
        targets = inputs.clone()
        inputs, loss_weight = self.mlm_prepare_data(inputs)
        outputs = self.forward(inputs)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.lr)
        # scheduler =
        return optimizer


class Replicator(pl.LightningModule):
    def __init__(self, conf: ReplicatorConfig):
        super(Replicator, self).__init__()
        self.replicator_network = ReplicatorNetwork(blocks_num=conf.blocks_num, seq_len=conf.seq_len,
                                                    embedding_size=conf.embedding_size, vocab_size=conf.vocab_size,
                                                    padding_idx=conf.padding_idx, mask=conf.mask)
        self.conf = conf

    def forward(self, x):
        return self.replicator_network(x)

    def one_step(self, batch, batch_idx):
        inputs = batch[:, :self.conf.seq_len]
        targets = batch[:, self.conf.predicted_num:self.conf.seq_len +
                        self.conf.predicted_num].clone().long()
        if self.conf.p1 > 0:
            inputs, loss_weight = mlm_prepare_data(inputs, vocab_size=self.conf.vocab_size, is_prefix=self.conf.is_prefix,
                                                   num_special_tokens=self.conf.num_special_tokens,
                                                   mask_token_id=self.conf.mask_token_id, p1=self.conf.p1,
                                                   p2=self.conf.p2, p3=self.conf.p3)
        outputs = self.forward(inputs)

        # --> (batch_size, max_sentence_len, vocab_size)
        # Exclude tokens where all probabilities degrade to 0
        tokens_probabilities_exist = torch.sum(outputs, dim=-1).bool()
        outputs = outputs[tokens_probabilities_exist, :]
        targets = targets[tokens_probabilities_exist]
        if self.conf.p1 > 0:
            loss_weight = loss_weight[tokens_probabilities_exist]
            loss = F.cross_entropy(
                outputs, targets, reduction='none') * loss_weight
            loss = loss.sum() / loss_weight.sum()
        else:
            loss = F.cross_entropy(outputs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.one_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.one_step(batch, batch_idx)
        self.log('valid_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        inputs = batch[:, :self.conf.seq_len]
        targets = inputs.clone()
        inputs, loss_weight = self.mlm_prepare_data(inputs)
        outputs = self.forward(inputs)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.lr)
        # scheduler =
        return optimizer
