import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from replicator.models.replicator import ReplicatorLayer, ReplicatorGPT
from replicator.datasets import FakeLMDataset

def test_replicator_gpt():
    """Replicator Model的输出不能是inf"""

    training_sentences_num = 128
    vocab_size = 100
    max_sentence_len = 64
    min_sentence_len = 64 // 4
    training_data = FakeLMDataset(sentences_num=training_sentences_num,
                    vocab_size=vocab_size, max_sentence_len=max_sentence_len, min_sentence_len=min_sentence_len)

    batch_size = 4
    embedding_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)


    inputs, targets, masks = next(iter(train_dataloader))

    layers_num = 4
    replicator_gpt = ReplicatorGPT(layers_num=layers_num, max_sentence_len=max_sentence_len,
                        vocab_size=vocab_size, embedding_size=embedding_size)

    loss = replicator_gpt(inputs, targets, masks)
    print(loss)
    assert not loss.isinf()

def test_replicator_layer():
    """
    如果输入是概率空间的一个分布，经过一步Replicator以后，输出也应该是概率空间的一个分布.
    在`embeding_size`充分大的情况下，经过一步Replicator以后，所有的概率都大于0.
    """

    training_sentences_num = 128
    vocab_size = 100
    max_sentence_len = 128
    min_sentence_len = 128 // 4
    training_data = FakeLMDataset(sentences_num=training_sentences_num,
                    vocab_size=vocab_size, max_sentence_len=max_sentence_len, min_sentence_len=min_sentence_len)

    batch_size = 4
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    embedding_size = 128
    # weight的N(0,1)分布是否合适，因为后面又softmax操作 ？
    embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)   

    softmax = nn.Softmax(dim=-1) 

    inputs, targets, masks = next(iter(train_dataloader))
    inputs_embedding = embedding(inputs)
    inputs_embedding[torch.logical_not(masks)] = float('-inf')
    inputs_embedding_softmax = torch.nan_to_num(softmax(inputs_embedding))

    replicatorLayer = ReplicatorLayer(max_sentence_len=max_sentence_len)

    outputs = replicatorLayer(inputs_embedding_softmax)

    # 确认经过一步Replicator之后，仍然是概率空间的分布
    assert torch.all(torch.abs(torch.sum(outputs[masks], dim=-1) - 1.) < 1e-5)
    # 确认经过一步Replicator之后，所有概率大于0
    print(outputs[outputs<0])
    assert list(outputs[outputs<0]) == []

