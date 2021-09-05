from typing import Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

@dataclass
class FakeLMDataset(Dataset):
    """
    Generate a fake dataset for Language Modeling. 
    The fake dataset is composed of a number of sentences. 
    Each sentence is composed of a number of tokens choosing from a vocabulary.
    The lengths of sentences are not exactly same with each other, so there are
    a maximum length and a minimum length for each sentence.

    All the tokens except the last token of each sentence are chosen as inputs of the model.
    All the tokens except the first token of each sentence are chosen as targets of the model.

    All the inputs and targets whose lengths are less than `max_sentence_len` are filled with
    the specific padding token which is 0.

    Arguments
    ---------
    sentences_num : int, number of sentences.
    vocab_size : int, vocabulary size.
    max_sentence_len : int, maximum sentence length.
    min_sentence_len : int, optional, minimum sentence length.
    
    Returns
    -------
    inputs : tensor with size (`max_sentence_len`,), inputs of the model.
    targets : tensor with size (`max_sentence_len`,), targets of the model.
    masks : tensor with size (`max_sentence_len`,), dtype=bool, 
            indicates whether there are padded with tokens, 0, at the corresponding indices.
    """
    def __init__(self, sentences_num: int, vocab_size: int, max_sentence_len: int, min_sentence_len: int=None):
        super().__init__()

        inputs = []
        targets = []
        masks = []

        tokens_num_predicted = 1  # the number of tokens predicted by the model

        if min_sentence_len is None:
            min_sentence_len = max_sentence_len // 4

        for i in range(sentences_num):
            # 1. Generate actual sentence length Randomly
            actual_sentence_len = int(torch.randint(min_sentence_len, max_sentence_len + 1, (1,)))
            # 2. Synthesize sentence by randomly choosing tokens from vocabulary
            #    Note: 0 is the padding token, here it can not be chosen.
            sentence = torch.randint(1, vocab_size, (actual_sentence_len + tokens_num_predicted,))
            # 3. Chose input and target of the model from the sentence, then fill up with padding token, i.e. 0 
            input = torch.cat([sentence[:-tokens_num_predicted], torch.zeros(max_sentence_len - actual_sentence_len)])
            target = torch.cat([sentence[tokens_num_predicted:], torch.zeros(max_sentence_len - actual_sentence_len)])
            mask = torch.cat([torch.ones_like(sentence[:-tokens_num_predicted]), torch.zeros(max_sentence_len - actual_sentence_len)])

            inputs += [input]
            targets += [target]
            masks += [mask]

        self.inputs = torch.stack(inputs).long()
        self.targets = torch.stack(targets).long()
        self.masks = torch.stack(masks).bool()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        mask = self.masks[index]
        return input, target, mask

class FakeLMDataModule(pl.LightningDataModule):
    def __init__(self, train_size: int, val_size: int, batch_size: int,
                 vocab_size: int, max_sentence_len: int, min_sentence_len: int=None):
        super().__init__()
        self.train_size, self.val_size, self.batch_size = train_size, val_size, batch_size
        self.vocab_size, self.max_sentence_len, self.min_sentence_len = vocab_size, max_sentence_len, min_sentence_len

    def setup(self):
        self.train_dataset = FakeLMDataset(sentences_num=self.train_size, vocab_size=self.vocab_size,
                             max_sentence_len=self.max_sentence_len, min_sentence_len=self.min_sentence_len)
        self.val_dataset = FakeLMDataset(sentences_num=self.val_size, vocab_size=self.vocab_size,
                             max_sentence_len=self.max_sentence_len, min_sentence_len=self.min_sentence_len)

    def train_dataloader(self):
         return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=True,
        )

