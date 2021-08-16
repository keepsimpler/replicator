from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

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

