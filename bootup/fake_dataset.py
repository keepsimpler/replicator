# %%
import torch

vocab_size, max_seq_len, items_num = 100, 10, 10
inputs = []
targets = []
# %%
act_seq_len = int(torch.randint(max_seq_len // 4, max_seq_len + 1, (1,)))
act_seq_len
# %%
sentence = torch.randint(1, vocab_size, (act_seq_len + 1,))
sentence
# %%
input = torch.cat([sentence[:-1], torch.zeros(max_seq_len - act_seq_len)])
input
# %%
target = torch.cat([sentence[1:], torch.zeros(max_seq_len - act_seq_len)])
target
# %%
mask = torch.cat([torch.ones_like(sentence[:-1]), torch.zeros(max_seq_len - act_seq_len)])
mask.bool()
# %%
inputs += [input]
targets += [target]
# %%
len(inputs), inputs
# %%
len(targets), targets
# %%
inputs = torch.stack(inputs)
# %%
inputs
# %%
tuple(inputs.size() )
# %%
