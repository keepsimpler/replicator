
from replicator.datasets import FakeLMDataset
from replicator.datasets.torchtext_wikitext2 import WikiText2Dataset

from torch.utils.data import DataLoader

def test_wikitext2_dataset():
    wikitext2 = WikiText2Dataset(batch_size=2, seq_len=64, split='train')
    wikitext2_iter = DataLoader(wikitext2, batch_size=None)
    inputs, targets, masks = next(iter(wikitext2_iter))
    assert tuple(inputs.size()) == (2, 64)
    assert tuple(targets.size()) == (2, 64)

def test_fake_lm_dataset():
    """确认Fake数据集的大小符合预期"""
    sentences_num = 10
    max_sentence_len = 10
    fakeLMDataset = FakeLMDataset(sentences_num=sentences_num, vocab_size=100, 
                                max_sentence_len=max_sentence_len, min_sentence_len=2)
    assert len(fakeLMDataset) == sentences_num
    input, target, mask = fakeLMDataset[0]
    assert tuple(input.size()) == (max_sentence_len,)
    assert tuple(target.size()) == (max_sentence_len,)
    assert tuple(mask.size()) == (max_sentence_len,)