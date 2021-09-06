# %%
from tqdm import tqdm
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
# %%
from transformers.file_utils import cached_path
dataset_file = cached_path("https://s3.amazonaws.com/datasets.huggingface.co/wikitext-103/wiki.train.tokens")
# %%
# dataset_file = 'data/wikitext103/wiki.train.tokens'

with open(dataset_file, "r", encoding="utf-8") as f:
    dataset = f.readlines()

# %%
dataset = list(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
                line.strip(' ').replace('\n', '[SEP]').replace('<unk>', '[UNK]'))) for line in tqdm(dataset))

# %%
dataset = torch.tensor([index for line in dataset for index in line], dtype=torch.long)
# %%
torch.save(dataset, 'data/wikitext103_train.tensor')
# %%
dataset = torch.load('data/wikitext103_train.tensor')
# %%
vocab_size = tokenizer.vocab_size
vocab_size
# %%
seq_len = 256
num_tokens = (dataset.size(0) // seq_len) * seq_len
seq_len, num_tokens
# %%
dataset = dataset.narrow(0, 0, num_tokens).view(-1, seq_len)
dataset.shape
# %%
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# %%
batch = next(iter(dataloader))
batch.shape
# %%
