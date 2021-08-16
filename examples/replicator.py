# %%
import torch
from torch.utils.data import DataLoader

from replicator.models import ReplicatorGPT
from replicator.datasets import FakeLMDataset

training_sentences_num = 2**16
vocab_size = 1000
max_sentence_len = 128
min_sentence_len = 128 // 4
training_data = FakeLMDataset(sentences_num=training_sentences_num,
                vocab_size=vocab_size, max_sentence_len=max_sentence_len, min_sentence_len=min_sentence_len)

batch_size = 8
embedding_size = 128
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

device = "cuda"

layers_num = 1
replicator_gpt = ReplicatorGPT(layers_num=layers_num, max_sentence_len=max_sentence_len,
                    vocab_size=vocab_size, embedding_size=embedding_size).to(device)

lr = 1e-1
optimizer = torch.optim.Adam(replicator_gpt.parameters(), lr=lr)

replicator_gpt.train()
for epoch in range(10):
    for inputs, targets, masks in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        loss = replicator_gpt(inputs, targets, masks)
        loss.backward()
        optimizer.step()
        print(f'epoch {epoch + 1}, loss {float(loss):.6f}')

# %%
