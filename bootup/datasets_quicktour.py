# %%
from datasets import load_dataset
# %%
dataset = load_dataset("data/wikitext.py", "wikitext-103-raw-v1", cache_dir='data')
# %%
dataset
# %%
dataset_train = dataset['train']
# %%
dataset_train[:10]
# %%
dataset_train.description
# %%
