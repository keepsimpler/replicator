# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from replicator.datasets import FakeLMDataset

# %%
training_sentences_num = 128
vocab_size = 100
max_sentence_len = 6  # 64
min_sentence_len = 4  # 64 // 4
training_data = FakeLMDataset(sentences_num=training_sentences_num,
                vocab_size=vocab_size, max_sentence_len=max_sentence_len, min_sentence_len=min_sentence_len)

batch_size = 4
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# %%
embedding_size = 3  # 64
# weight的N(0,1)分布是否合适，因为后面又softmax操作 ？
embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
# %%
embedding_data = embedding(training_data[0][0])
# %%
training_data[0][0]
# %%
softmax = nn.Softmax(dim=-1)
# %%
softmax(embedding_data)
# %%
for inputs, targets, masks in train_dataloader:
    inputs_embedding = embedding(inputs)
    inputs_embedding[torch.logical_not(masks)] = float('-inf')
    inputs_embedding_softmax = torch.nan_to_num(softmax(inputs_embedding))

# %%
inputs, masks
# %%
inputs_embedding
# %%
inputs_embedding_softmax
# %%
tuple(inputs_embedding_softmax.shape) == (batch_size, max_sentence_len, embedding_size)
# %%
inputs = inputs_embedding_softmax[0:2]
masks = masks[0:2]
targets = targets[0:2]
inputs.shape, masks.shape
# %%
# m --> max_sentence_len, n --> embedding_size
# inputs = torch.einsum('... m n -> ... n m', inputs)
# %%
queries_weight = torch.randn(max_sentence_len, max_sentence_len)
keys_weight = torch.randn(max_sentence_len, max_sentence_len)

# repeated `max_sentence_len` times
queries_weight_repeated = queries_weight.repeat(max_sentence_len, 1, 1)
for i in range(max_sentence_len):
    queries_weight_repeated[i, :, i+1:] = 0.
    queries_weight_repeated[i, i+1:, :] = 0.

keys_weight_repeated = keys_weight.repeat(max_sentence_len, 1, 1)
for i in range(max_sentence_len):
    keys_weight_repeated[i, :, i+1:] = 0.
    keys_weight_repeated[i, i+1:, :] = 0.

# %%
# b --> batch_size, m --> embedding_size, n --> max_sentence_len,  p --> max_sentence_len
queries = torch.einsum('b n m, ... n p -> b ... m p', inputs, queries_weight_repeated)
queries.shape, queries
# %%
# b --> batch_size, m --> embedding_size, n --> max_sentence_len,  p --> max_sentence_len
keys = torch.einsum('b n m, ... n p -> b ... p m', inputs, keys_weight_repeated)
keys.shape, keys
# %%
# m --> embedding_size, n --> max_sentence_len, q --> embedding_size
attentions = torch.einsum('... m n, ... n q -> ... m q', queries, keys)
attentions.shape, attentions
# %%
inputs.shape, queries.shape, keys.shape, attentions.shape
# %%
# compute Ax, to get fitnesses
# m --> embedding_size, n --> max_sentence_len, p --> embedding_size,
fitnesses = torch.einsum('... n m p, b n p -> ... n m', attentions, inputs)
fitnesses.shape, fitnesses
# %%
# compute x * Ax, then summarize through the `embedding_size` dim, in order to get average fitness
# m --> embedding_size
avg_fitness = torch.einsum('... m, ... m -> ...', inputs, fitnesses)
avg_fitness.shape, avg_fitness
# %%
# unsqueeze the `embedding_size` dim
avg_fitness = avg_fitness.unsqueeze(-1)
avg_fitness.shape
# %%
# compute Ax - x * Ax, to get the net fitnesses
net_fitnesses = fitnesses - avg_fitness
net_fitnesses.shape, net_fitnesses
# %%
# compute derivative of inputs
inputs_derivate = inputs * net_fitnesses
inputs_derivate.shape, inputs_derivate
# %%
# Eluer method of ode
inputs_next = inputs + inputs_derivate
inputs_next
# %%
torch.sum(inputs_next, dim=-1)
# %%
list(inputs_next[inputs_next<0]) == []
torch.all(torch.abs(torch.sum(inputs_next[masks], dim=-1) - 1.) < 1e-5)

# %%
inputs_next.shape

# %%
linear = nn.Linear(embedding_size, vocab_size, bias=False)
# %%
logit = linear(inputs_next)
logit = logit.permute(0,2,1)
# %%
targets
# %%
loss = nn.CrossEntropyLoss(ignore_index=0)
# %%
loss(logit, targets)
# %%
