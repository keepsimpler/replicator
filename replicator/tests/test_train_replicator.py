import torch
import torch.nn as nn

from replicator.models.replicator import ReplicatorGPT
from replicator.datasets.torchtext_wikitext2 import WikiText2Dataset
from replicator.trainings.train_replicator import Trainer


def test_train_replicator():
    batch_size = 16
    max_sentence_len = 64
    wikitext2_train = WikiText2Dataset(batch_size=batch_size, seq_len=max_sentence_len, split='train')

    vocab_size = wikitext2_train.vocab_size()
    embedding_size = 128
    blocks_num = 4
    replicator_gpt = ReplicatorGPT(blocks_num=blocks_num, max_sentence_len=max_sentence_len,
                    vocab_size=vocab_size, embedding_size=embedding_size, mask=True)

    lr = 1e-2
    optimizer = torch.optim.Adam(replicator_gpt.parameters(), lr=lr)

    # num_epochs = 1

    # trainer = Trainer(replicator_gpt, wikitext2_train, optimizer)
    # trainer.train(num_epochs)