from typing import Optional

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import pytorch_lightning as pl


class WikiText2Dataset(IterableDataset):
    """"""
    def __init__(self, batch_size: int, seq_len: int, split: str) -> None:
        super().__init__()
        raw_text_iter = WikiText2(split=split)
        print(len(raw_text_iter))
        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(map(tokenizer, raw_text_iter), specials=['<pad>', '<unk>'])
        vocab.set_default_index(vocab['<pad>'])

        # raw_text_iter was "consumed" by the process of building the vocab,
        # so we have to create it again
        raw_text_iter = WikiText2(split=split) 
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in iter(raw_text_iter)]
        data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

        sequence_dim = data.size(0) // batch_size
        data = data[:sequence_dim * batch_size]
        self.data = data.view(batch_size, sequence_dim).contiguous()
        self.sequence_num = sequence_dim // seq_len
        self.seq_len = seq_len
        self.vocab = vocab

        self.masks = torch.ones(batch_size, seq_len).bool()

    def __iter__(self):
        for i in range(self.sequence_num):
            yield self.data[:, i*self.seq_len:(i+1)*self.seq_len], \
                  self.data[:, i*self.seq_len+1:(i+1)*self.seq_len+1], \
                  self.masks

    # def __len__(self):
    #     return self.sequence_num

    def vocab_size(self):
        return len(self.vocab)

class WikiText2DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, seq_len: int):
        super().__init__()
        self.batch_size, self.seq_len = batch_size, seq_len

    def setup(self):
        self.train_data = WikiText2Dataset(batch_size=self.batch_size, seq_len=self.seq_len, split='train')
        self.val_data = WikiText2Dataset(batch_size=self.batch_size, seq_len=self.seq_len, split='valid')
        self.vocab_size = self.train_data.vocab_size()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=None
        )



if __name__ == '__main__':
    wikitext2 = WikiText2Dataset(batch_size=2, seq_len=10, split='train')
    print(len(wikitext2))
    x, targets = next(iter(wikitext2))
    print(x.shape, targets.shape)
    print(wikitext2.vocab_size())