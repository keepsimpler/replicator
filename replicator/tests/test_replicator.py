import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from replicator.models.replicator import uniformal_matrix, ReplicatorLayer, StochasticEmbedding, StochasticProjection, UniformConfig, NormalConfig
from replicator.models.replicator import ReplicatorGPT
from replicator.datasets import FakeLMDataset

def test_uniformal_matrix():
    "当均匀分布矩阵足够大时，它的最大元素的值应该逼近均匀分布上限值，最小元素的值应该逼近均匀分布下限值"
    uniformConfig = UniformConfig(dim1=100, dim2=100, scale=2)
    m = uniformal_matrix(uniformConfig)
    assert 2 - 1e-2 <= torch.max(m) <= 2
    assert -2 <= torch.min(m) <= -2 + 1e-2

def test_stochastic_embedding():
    ""
    x = torch.tensor([[2,1,4,5], [5,8,3,7]])
    uniformConfig = UniformConfig(dim1=10, dim2=3, scale=2)
    stochastic_embedding = StochasticEmbedding(uniformConfig)
    y = stochastic_embedding(x)
    # 确认变换之后，该维度所有元素的和等于1
    assert torch.all(torch.abs(torch.sum(y, dim=-1) - 1.) < 1e-5)

def test_stochastic_embedding_padding_idx():
    "Index Tensor中值等于0的元素embedding into all zeros"
    x = torch.tensor([[2, 1, 4, 0, 0], [5, 8, 3, 7, 0]])
    uniformConfig = UniformConfig(dim1=10, dim2=3, scale=2)
    stochastic_embedding = StochasticEmbedding(uniformConfig)
    y = stochastic_embedding(x)
    # 确认对应位置的embedding等于zero
    assert torch.all(y[0, 3:, :] == 0.) and torch.all(y[1, 4, :] == 0.)

def test_stochastic_projection():
    ""
    x = torch.tensor([[2,1,4,5], [5,8,3,7]])
    uniformConfig = UniformConfig(dim1=10, dim2=3, scale=2)
    stochastic_embedding = StochasticEmbedding(uniformConfig)
    y = stochastic_embedding(x)
    # 确认变换之后，该维度所有元素的和等于1
    assert torch.all(torch.abs(torch.sum(y, dim=-1) - 1.) < 1e-5)

    stochastic_projection = StochasticProjection(uniformConfig)
    z = stochastic_projection(y)
    assert z.shape[-1] == 10
    # 确认变换之后，该维度所有元素的和等于1
    # assert torch.all(torch.abs(torch.sum(z, dim=-1) - 1.) < 1e-5)

def test_replicator_gpt_with_backward():
    x = torch.tensor([[2,1,4,5], [5,8,3,7]])
    targets = torch.tensor([[1,4,5,7], [8,3,7,1]])
    masks = torch.tensor([[True,True,True,True],[True,True,True,True]])
    replicator_gpt = ReplicatorGPT(blocks_num=1, max_sentence_len=4, vocab_size=10, embedding_size=3, mask=True)
    loss = replicator_gpt(x, targets, masks)
    loss.backward()
    # for name, param in replicator_gpt.named_parameters():
    #     if param.grad is not None:
    #         print(param.data.sign(), param.grad.sign())
    print(loss)