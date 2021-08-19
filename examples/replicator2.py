# %%
import torch
from torch.utils.data import DataLoader

from replicator.models.replicator2 import ReplicatorGPT
from replicator.datasets import FakeLMDataset

training_sentences_num = 2**16
vocab_size = 24
max_sentence_len = 16
min_sentence_len = 16 // 4
training_data = FakeLMDataset(sentences_num=training_sentences_num,
                vocab_size=vocab_size, max_sentence_len=max_sentence_len, min_sentence_len=min_sentence_len)

batch_size = 2
embedding_size = 8
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# %%
device = "cpu"

blocks_num = 2
replicator_gpt = ReplicatorGPT(blocks_num=blocks_num, max_sentence_len=max_sentence_len,
                    vocab_size=vocab_size, embedding_size=embedding_size, mask=True).to(device)

lr = 1e-3
optimizer = torch.optim.Adam(replicator_gpt.parameters(), lr=lr)

replicator_gpt.train()
for epoch in range(10):
    for inputs, targets, masks in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        loss = replicator_gpt(inputs, targets)
        # with torch.autograd.set_detect_anomaly(True):
        loss.backward()
        optimizer.step()
        print(f'epoch {epoch + 1}, loss {float(loss):.6f}')

# %%
