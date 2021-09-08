# %%
import torch
import pytorch_lightning as pl

from replicator.models.replicator import ReplicatorGPT
from replicator.datasets.torchtext_wikitext2 import WikiText2DataModule
from replicator.datasets.wikitext103 import WikiText103DataModule

# %% wikitext2
batch_size = 16
seq_len = 64

# %%
data = WikiText2DataModule(batch_size=batch_size, seq_len=seq_len)
data.setup()
data_loader = data.train_dataloader()
# %%
inputs, targets, masks = next(iter(data_loader))
inputs.shape
# %%
vocab_size = data.train_data.vocab_size()
vocab_size
# %%
embedding_size = 128
blocks_num = 4
model = ReplicatorGPT(blocks_num=blocks_num,max_sentence_len=seq_len,
                    vocab_size=vocab_size,embedding_size=embedding_size, lr = 1e-2)
# %%
trainer = pl.Trainer(gpus=1, max_epochs = 5)
# %%
trainer.fit(model, datamodule=data)
# %%
# lr_finder = trainer.tuner.lr_find(model, datamodule=data)
# lr_finder.suggestion()
# fig = lr_finder.plot(suggest=True)
# fig.show()
# %%
trainer.validate(model, datamodule=data)

# %%
trainer_wikitext2 = pl.Trainer(gpus=1, max_epochs = 6, resume_from_checkpoint="lightning_logs/version_1/checkpoints/epoch=4-step=10004.ckpt")
# %%
trainer_wikitext2.fit(model, datamodule=data)
# %%
trainer_wikitext2.validate(model, datamodule=data)
# %%
prompt = "as a child"

# %%
vocab = data.train_data.vocab
# %%
vocab["child"], len(vocab)
# %%
vocab(["as", "a", "child"])
# %%
vocab.get_default_index()
# %%
vocab.lookup_tokens([15,9,863])
# %%
prompt = "Barker was born the second daughter and youngest child of Walter Barker . in"
prompt.split()
# %%
# ["as", "a", "child", "of", ",", "of", "that", "of", "months", "by", "announced", "by","<unk>"]
input_ids = vocab(prompt.split())
input_ids = torch.tensor(input_ids)
input_ids = torch.cat([input_ids, torch.zeros(seq_len - input_ids.size(0))]).unsqueeze(0).long()
input_ids.shape
# %%
masks = (input_ids != 0)
masks.shape, masks.dtype
# %%
logits = model(input_ids, masks)
logits.shape
# %%
predicts = logits.argmax(dim=2).flatten().tolist()
predicts
# %%
vocab.lookup_tokens(predicts)


# %% Wikitext103
batch_size = 16
seq_len = 512
data = WikiText103DataModule(batch_size=batch_size, seq_len=seq_len, data_dir='data')
data.setup()
train_dataloader = data.train_dataloader()
# %%
inputs, targets, masks = next(iter(train_dataloader))
inputs.shape, targets.shape, masks.shape

# %%
vocab_size = data.vocab_size
print('vocab_size:', data.vocab_size)
# %%
embedding_size = 512
blocks_num = 12
lr = 1e-2
model = ReplicatorGPT(blocks_num=blocks_num,max_sentence_len=seq_len-1,
                    vocab_size=vocab_size,embedding_size=embedding_size, lr = lr)
# %%
trainer_wikitext103 = pl.Trainer(gpus=1, max_epochs = 3, resume_from_checkpoint="lightning_logs/version_0/checkpoints/epoch=2-step=43310.ckpt")

# %%
trainer_wikitext103.validate(model, datamodule=data)
# %%
