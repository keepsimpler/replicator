import os
from tqdm import tqdm
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers.file_utils import cached_path

import pytorch_lightning as pl

def shift_one_token(batch):
    batch = torch.stack(batch)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    masks = torch.ones_like(inputs).bool()
    return inputs, targets, masks


class WikiText103DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, seq_len: int):
        super().__init__()
        self.batch_size, self.seq_len = batch_size, seq_len

    def setup(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        # if train_data exists
        if os.path.isfile('data/wikitext103.train.tensor'):
            train_data = torch.load('data/wikitext103.train.tensor')
        else:  # else, download and construct train_data
            train_data_file = cached_path("https://s3.amazonaws.com/datasets.huggingface.co/wikitext-103/wiki.train.tokens")
            with open(train_data_file, "r", encoding="utf-8") as f:
                train_data = f.readlines()
            train_data = list(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
                line.strip(' ').replace('\n', '[SEP]').replace('<unk>', '[UNK]'))) for line in tqdm(train_data))
            train_data = torch.tensor([index for line in train_data for index in line], dtype=torch.long)
            torch.save(train_data, 'data/wikitext103.train.tensor')

        num_tokens = (train_data.size(0) // self.seq_len) * self.seq_len
        self.train_data = train_data.narrow(0, 0, num_tokens).view(-1, self.seq_len)

        # if valid_data exists
        if os.path.isfile('data/wikitext103.valid.tensor'):
            valid_data = torch.load('data/wikitext103.valid.tensor')
        else:  # else, download and construct valid_data
            valid_data_file = cached_path("https://s3.amazonaws.com/datasets.huggingface.co/wikitext-103/wiki.valid.tokens")
            with open(valid_data_file, "r", encoding="utf-8") as f:
                valid_data = f.readlines()
            valid_data = list(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
                line.strip(' ').replace('\n', '[SEP]').replace('<unk>', '[UNK]'))) for line in tqdm(valid_data))
            valid_data = torch.tensor([index for line in valid_data for index in line], dtype=torch.long)
            torch.save(valid_data, 'data/wikitext103.valid.tensor')

        num_tokens = (valid_data.size(0) // self.seq_len) * self.seq_len
        self.valid_data = valid_data.narrow(0, 0, num_tokens).view(-1, self.seq_len)

        self.vocab_size = tokenizer.vocab_size

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=shift_one_token
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=shift_one_token
        )

