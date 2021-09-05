# %%
import torch
import pytorch_lightning as pl

from replicator.models.replicator import ReplicatorGPT
from replicator.datasets.torchtext_wikitext2 import WikiText2DataModule

# %%
batch_size = 16
seq_len = 64

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
trainer = pl.Trainer(gpus=1, max_epochs = 5, auto_lr_find=False)
# %%
# trainer.tune(model, datamodule=data)

# %%
trainer.fit(model, datamodule=data)
# %%
lr_finder = trainer.tuner.lr_find(model, datamodule=data)
# %%
lr_finder.suggestion()
# %%
fig = lr_finder.plot(suggest=True)
fig.show()
# %%
model.hparams.lr = lr_finder.suggestion
# %%
