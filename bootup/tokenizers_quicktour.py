# %%
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# %%
from tokenizers.trainers import BpeTrainer

# vocab_size=20000, min_frequency=2,
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
# %%
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
# %%
files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
tokenizer.train(files, trainer)
# %%
tokenizer.save("data/tokenizer-wiki.json")
# %%
tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")
# %%
sentence = "Hello, y'all! How are you 😁 ?"
output = tokenizer.encode(sentence)
# %%
output.tokens
# %%
output.ids
# %%
output.offsets[10], sentence[26:27]
# %%
tokenizer.token_to_id("[PAD]")
# %%
# from tokenizers.processors import TemplateProcessing

# tokenizer.post_processor = TemplateProcessing(
#     single="[CLS] $A [SEP]",
#     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
#     special_tokens=[
#         ("[CLS]", tokenizer.token_to_id("[CLS]")),
#         ("[SEP]", tokenizer.token_to_id("[SEP]")),
#     ]
# )

# %%
output = tokenizer.encode("Hello, y'all!", "How are you 😁 ?")
output.tokens
# %%
output.type_ids


# %%
from datasets import load_dataset
# %%
dataset = load_dataset("data/wikitext.py", "wikitext-103-raw-v1")
# %%
dataset
# %%
dataset_train = dataset['train']
# %%
dataset_train[0]['text']
# %%
dataset_train.description
# %%
def encode(item):
    output = tokenizer.encode(item['text'])
    return {
        'tokens': output.tokens,
        'ids': output.ids
    }

dataset_train = dataset_train.map(encode, batched=False)
# %%
dataset_train
# %%
import torch
# %%
dataset_train.set_format(type="torch", columns=["ids"])
# %%
dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=32)
# %%
next(iter(dataloader))
# %%
dataset_train.features
# %% [markdown]
## some comment
# %%
