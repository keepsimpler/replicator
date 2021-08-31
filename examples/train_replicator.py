# %%
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import wikitext2

from replicator.models.replicator import ReplicatorGPT
from replicator.models.pytorch_transformer import TransformerModel, generate_square_subsequent_mask

from replicator.datasets import FakeLMDataset
from replicator.utils import Accumulator

# %%  fake dataset
training_sentences_num = 2**16
validate_sentences_num = training_sentences_num // 2**6

vocab_size = 2*8
max_sentence_len = 16
min_sentence_len = 16 // 4
training_data = FakeLMDataset(sentences_num=training_sentences_num,
                vocab_size=vocab_size, max_sentence_len=max_sentence_len, min_sentence_len=min_sentence_len)
validate_data = FakeLMDataset(sentences_num=validate_sentences_num,
                vocab_size=vocab_size, max_sentence_len=max_sentence_len, min_sentence_len=min_sentence_len)

batch_size = 16
embedding_size = 16
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validate_dataloader = DataLoader(validate_data, batch_size=batch_size, shuffle=True)

# %% wikitext2
from replicator.datasets.torchtext_wikitext2 import WikiText2Dataset

batch_size = 16
max_sentence_len = 64
wikitext2_train = WikiText2Dataset(batch_size=batch_size, seq_len=max_sentence_len, split='train')
vocab_size = wikitext2_train.vocab_size()
embedding_size = 128
wikitext2_val = WikiText2Dataset(batch_size=batch_size, seq_len=max_sentence_len, split='valid')

# %%
def evaluate_perplexity(net, validate_dataloader):
    net.eval()
    metric = Accumulator(2)
    for inputs, targets, masks in validate_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        loss = replicator_gpt(inputs, targets, masks)
        metric.add(loss, 1)
    loss = metric[0] / metric[1]
    perplexity = math.exp(metric[0] / metric[1])
    return perplexity, loss

# %%
from accelerate import Accelerator

device = "cpu"
accelerator = Accelerator()
print(accelerator.device)

# %% fake dataset training using replicator_gpt
blocks_num = 4
replicator_gpt = ReplicatorGPT(blocks_num=blocks_num, max_sentence_len=max_sentence_len,
                    vocab_size=vocab_size, embedding_size=embedding_size, mask=True).to(device)

lr = 1e-2
optimizer = torch.optim.Adam(replicator_gpt.parameters(), lr=lr)

num_epochs = 5

# %%
from replicator.trainings.train_replicator import Trainer

trainer = Trainer(replicator_gpt, wikitext2_train, optimizer)
trainer.train(num_epochs)
# %%
replicator_gpt, optimizer, train_dataloader, validate_dataloader = accelerator.prepare(
    replicator_gpt, optimizer, train_dataloader, validate_dataloader)

replicator_gpt.train()
for epoch in range(num_epochs):
    replicator_gpt.train()
    epoch_metric = Accumulator(2)
    for inputs, targets, masks in train_dataloader:
        # inputs = inputs.to(device)
        # targets = targets.to(device)
        # masks = masks.to(device)
        optimizer.zero_grad()
        loss = replicator_gpt(inputs, targets, masks)
        # with torch.autograd.set_detect_anomaly(True):
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        with torch.no_grad():
            epoch_metric.add(loss, 1)
    loss = epoch_metric[0] / epoch_metric[1]
    perplexity = math.exp(epoch_metric[0] / epoch_metric[1])
    print(f'epoch {epoch + 1}, perplexity {float(perplexity):.6f}, loss {float(loss):.6f}')
    eval_perplexity, eval_loss = evaluate_perplexity(replicator_gpt, validate_dataloader)
    print(f'epoch {epoch + 1}, eval_perplexity {float(eval_perplexity):.6f}, eval_loss {float(eval_loss):.6f}')

# %% wikitext2 training with replicator_gpt 
blocks_num = 2
replicator_gpt = ReplicatorGPT(blocks_num=blocks_num, max_sentence_len=max_sentence_len,
                    vocab_size=vocab_size, embedding_size=embedding_size, mask=True).to(device)

lr = 1e-3
optimizer = torch.optim.Adam(replicator_gpt.parameters(), lr=lr)

num_epochs = 5

replicator_gpt.train()
for epoch in range(num_epochs):
    replicator_gpt.train()
    epoch_metric = Accumulator(2)
    for inputs, targets, masks in wikitext2_train:
        # masks = torch.ones_like(inputs).bool()
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        loss = replicator_gpt(inputs, targets, masks)
        # with torch.autograd.set_detect_anomaly(True):
        loss.backward()
        print(loss)
        # accelerator.backward(loss)
        optimizer.step()
        with torch.no_grad():
            epoch_metric.add(loss, 1)
    loss = epoch_metric[0] / epoch_metric[1]
    perplexity = math.exp(epoch_metric[0] / epoch_metric[1])
    print(f'epoch {epoch + 1}, perplexity {float(perplexity):.6f}, loss {float(loss):.6f}')
    # eval_perplexity, eval_loss = evaluate_perplexity(replicator_gpt, validate_dataloader)
    # print(f'epoch {epoch + 1}, eval_perplexity {float(eval_perplexity):.6f}, eval_loss {float(eval_loss):.6f}')

# %% wikitext2 training with transformer 
d_hid = embedding_size # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability

model = TransformerModel(max_sentence_len, vocab_size, embedding_size, nhead, d_hid, nlayers, dropout).to(device)

lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model, optimizer, wikitext2_train, wikitext2_val = accelerator.prepare(
    model, optimizer, wikitext2_train, wikitext2_val)

# %%

criterion = nn.CrossEntropyLoss()

num_epochs = 5

# src_mask = generate_square_subsequent_mask(max_sentence_len)

model.train()
for epoch in range(num_epochs):
    model.train()
    epoch_metric = Accumulator(2)
    for inputs, targets, masks in wikitext2_train:
        inputs = inputs.to(accelerator.device)
        targets = targets.to(accelerator.device)
        optimizer.zero_grad()
        loss = model(inputs, targets)
        # loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
        print(loss)
        # with torch.autograd.set_detect_anomaly(True):
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        with torch.no_grad():
            epoch_metric.add(loss, 1)
    loss = epoch_metric[0] / epoch_metric[1]
    perplexity = math.exp(epoch_metric[0] / epoch_metric[1])
    print(f'epoch {epoch + 1}, perplexity {float(perplexity):.6f}, loss {float(loss):.6f}')
    # eval_perplexity, eval_loss = evaluate_perplexity(model, validate_dataloader)
    # print(f'epoch {epoch + 1}, eval_perplexity {float(eval_perplexity):.6f}, eval_loss {float(eval_loss):.6f}')
# %%
