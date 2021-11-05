import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import pytorch_lightning as pl

class ReplicatorLayer(nn.Module):
  def __init__(self, prob_space_size: int, mask: bool=False):
    super(ReplicatorLayer, self).__init__()
    weight = torch.Tensor(prob_space_size, prob_space_size)
    nn.init.xavier_uniform_(weight)
    if mask:
      mask_matrix = torch.triu(torch.ones(prob_space_size, prob_space_size, dtype=torch.bool), diagonal=1)
      weight.masked_fill_(mask_matrix, 0)
    self.weight = Parameter(weight)

  def forward(self, x):
    # 1. compute Wx, matrix-vector product between W and x
    fitnesses = torch.einsum('m n, i j n -> i j m', self.weight, x)
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
  def __init__(self, seq_len: int, embedding_size: int, mask: bool=False):
    super(ReplicatorBlock, self).__init__()
    self.replicator_embedding = ReplicatorLayer(prob_space_size=embedding_size)
    self.replicator_seq = ReplicatorLayer(prob_space_size=seq_len, mask=mask)
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


class ReplicatorGPT(pl.LightningModule):
  def __init__(self, blocks_num: int, seq_len: int, vocab_size: int, embedding_size: int, predicted_num: int=1,
                lr: float=5e-3, mask: bool=True):
    super(ReplicatorGPT, self).__init__()
    replicator_blocks = [ReplicatorBlock(seq_len, embedding_size, mask) for _ in range(blocks_num)]
    self.replicator_blocks = nn.Sequential(*replicator_blocks)

    self.projection = Projection(embedding_size, vocab_size)
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.softmax = nn.Softmax(dim=-1)
    self.seq_len, self.predicted_num = seq_len, predicted_num
    self.lr = lr

    self.kl_div = nn.KLDivLoss(reduction="batchmean")

  def forward(self, x):
    x = self.embedding(x)
    x_prob = self.softmax(x)
    y_prob = self.replicator_blocks(x_prob)
    kl_div_loss = self.kl_div(x_prob.log(), y_prob)
    self.log('kl_div_loss', kl_div_loss)
    y = self.projection(y_prob)
    return y

  def one_step(self, batch, batch_idx):
    inputs = batch[:, :self.seq_len]
    targets = batch[:, self.predicted_num:self.seq_len + self.predicted_num].long()
    outputs = self.forward(inputs)

    # --> (batch_size, max_sentence_len, vocab_size)
    # Exclude tokens where all probabilities degrade to 0
    tokens_probabilities_exist = torch.sum(outputs, dim=-1).bool()
    outputs = outputs[tokens_probabilities_exist, :]
    targets = targets[tokens_probabilities_exist]
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
    inputs = batch[:, :self.seq_len]
    targets = batch[:, self.predicted_num:self.seq_len + self.predicted_num].long()
    outputs = self.forward(inputs)
    return outputs

  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
      # scheduler = 
      return optimizer


class MLMPrepareData(nn.Module):
  """
  Prepare data for MLM (mask language model).
  Inputs have shape (batch_size, seq_len)
  Arguments:
  ----------
  batch_size and seq_len: shape of inputs
  vocab_size: default is 50258(50256 tokens + 1 <endoftext> token + 1(the <mask> token)
  min_special_token_id: default is 50256 represent the <endoftext> special tokens
  mask_token_id: default is 50257 represent the <mask> token
  p1: probability of tokens that will be selected at random as masked tokens
  p2: probability of replaced by s special <mask> token or a random token for a masked token
  p3: probability of replaced by a random token for the token replaced by a spceical <mask> token or random token
  """
  def __init__(self, batch_size, seq_len: int, vocab_size: int=50258, min_special_token_id: int=50256, 
               mask_token_id: int=50257, p1: float=0.15, p2: float=0.9, p3: float=1/9):
    super(MLMPrepareData, self).__init__()
    input_masks_1 = torch.rand(batch_size, seq_len) < p1
    self.register_buffer('input_masks_1', input_masks_1)
    input_masks_2 = input_masks_1 & (torch.rand(batch_size, seq_len) < p2)
    self.register_buffer('input_masks_2', input_masks_2)
    input_masks_3 = input_masks_2 & (torch.rand(batch_size, seq_len) < p3)
    self.register_buffer('input_masks_3', input_masks_3)

    random_tokens = torch.randint(0, vocab_size - 2, (input_masks_3.sum().item(),), dtype=torch.int32)
    self.register_buffer('random_tokens', random_tokens)

    loss_weight = torch.zeros(batch_size, seq_len)
    loss_weight[input_masks_1] = 1
    self.register_buffer('loss_weight', loss_weight)

    self.min_special_token_id = min_special_token_id ## min special token id
    self.mask_token_id = mask_token_id

  def forward(self, inputs):
    self.input_masks_2[inputs >= self.min_special_token_id] = False  # Do not mask special tokens
    inputs[self.input_masks_2] = self.mask_token_id
    inputs[self.input_masks_3] = self.random_tokens

    # self.loss_weight[self.input_masks_1] = 1
    return inputs, self.loss_weight


class ReplicatorBert(pl.LightningModule):
  def __init__(self, blocks_num: int, batch_size: int, seq_len: int, embedding_size: int, vocab_size: int=50258,
                min_special_token_id: int=50256, mask_token_id: int=50257, 
                p1: float=0.15, p2: float=0.9, p3: float=1/9, lr: float=5e-3):
    super(ReplicatorBert, self).__init__()

    self.mlm_prepare_data = MLMPrepareData(batch_size, seq_len, vocab_size, min_special_token_id,
                                           mask_token_id, p1, p2, p3)

    replicator_blocks = [ReplicatorBlock(seq_len, embedding_size, mask=False) for _ in range(blocks_num)]
    self.replicator_blocks = nn.Sequential(*replicator_blocks)

    self.projection = Projection(embedding_size, vocab_size)
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.softmax = nn.Softmax(dim=-1)
    self.seq_len = seq_len
    self.lr = lr

    self.kl_div = nn.KLDivLoss(reduction="batchmean")

  def forward(self, x):
    x = self.embedding(x)
    x_prob = self.softmax(x)
    y_prob = self.replicator_blocks(x_prob)
    kl_div_loss = self.kl_div(x_prob.log(), y_prob)
    self.log('kl_div_loss', kl_div_loss)
    y = self.projection(y_prob)
    return y

  def one_step(self, batch, batch_idx):
    inputs = batch[:, :self.seq_len]
    targets = inputs.clone().long()
    inputs, loss_weight = self.mlm_prepare_data(inputs)
    outputs = self.forward(inputs)

    # --> (batch_size, max_sentence_len, vocab_size)
    # Exclude tokens where all probabilities degrade to 0
    tokens_probabilities_exist = torch.sum(outputs, dim=-1).bool()
    outputs = outputs[tokens_probabilities_exist, :]
    targets = targets[tokens_probabilities_exist]
    loss_weight = loss_weight[tokens_probabilities_exist]
    loss = F.cross_entropy(outputs, targets, reduction='none') * loss_weight
    loss = loss.sum() / loss_weight.sum()
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
    inputs = batch[:, :self.seq_len]
    targets = inputs.clone()
    inputs, loss_weight = self.mlm_prepare_data(inputs)
    outputs = self.forward(inputs)
    return outputs

  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
      # scheduler = 
      return optimizer
