# %%
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm

# %%
class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
# %%
d_model = 768
conv1d = Conv1D(d_model, d_model*3)
# represents a sequence of batch_size=1, seq_len=4 and embedding_sz=768,
# something like "Hello how are you"
x = torch.rand(1, 4, d_model)
x = conv1d(x)
x.shape
# %%
query, key, value = x.split(d_model, dim=-1)
query.shape, key.shape, value.shape
# %%
class FeedForward(nn.Module):
    def __init__(self, dropout, d_model=768, nx=768*4):
        super().__init__()
        self.c_fc = Conv1D(d_model, nx)
        self.c_proj = Conv1D(nx, d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))

# %%
# class Attention(nn.Module):
#     def __init__(self, d_model=768, n_head=12, n_ctx=1024, d_head=64, bias=True, scale=False):
#         super().__init__()
#         self.n_head  = n_head
#         self.d_model = d_model
#         self.c_attn  = Conv1D(d_model, d_model*3)
#         self.scale   = scale
#         self.softmax = nn.Softmax(dim=-1)
#         self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
#         self.dropout = nn.Dropout(0.1)
#         self.c_proj  = Conv1D(d_model, d_model)
        
#     def split_heads(self, x):
#         "return shape [`batch`, `head`, `sequence`, `features`]"
#         new_shape = x.size()[:-1] + (self.n_head, x.size(-1)//self.n_head) 
#         x = x.view(*new_shape)
#         return x.permute(0, 2, 1, 3) 
    
#     def _attn(self, q, k, v, attn_mask=None):
#         scores  = torch.matmul(q, k.transpose(-2, -1))
#         if self.scale: scores = scores/math.sqrt(v.size(-1))
#         nd, ns  = scores.size(-2), scores.size(-1)
#         if attn_mask is not None: scores = scores + attn_mask
#         scores  = self.softmax(scores)
#         scores  = self.dropout(scores)
#         outputs = torch.matmul(scores, v)
#         return outputs
    
#     def merge_heads(self, x):
#         x         = x.permute(0, 2, 1, 3).contiguous()
#         new_shape = x.size()[:-2] + (x.size(-2)*x.size(-1),)
#         return x.view(*new_shape)
        
#     def forward(self, x):
#         x        = self.c_attn(x) #new `x` shape - `[1,3,2304]`
#         q, k, v  = x.split(self.d_model, dim=2)
#         q, k, v  = self.split_heads(q), self.split_heads(k), self.split_heads(v)
#         out      = self._attn(q, k, v)
#         out      = self.merge_heads(out)
#         out      = self.c_proj(out)
#         return out
# %%
class Attention(nn.Module): # from _FASTAI
    def __init__(self, d_model=768, n_head=12, d_head=64, n_ctx=1024, bias=True, scale=False):
        super().__init__()
        self.n_head   = n_head
        self.d_head   = d_head
        self.softmax  = nn.Softmax(dim=-1)
        self.scale    = scale
        self.atn_drop = nn.Dropout(0.1)
        self.wq, self.wk, self.wv = [nn.Linear(d_model, n_head*d_head,
                                               bias=bias) for o in range(3)]
   
    def split_heads(self, x, layer, bs):
        x = layer(x)
        return x.view(bs, x.size(1), self.n_head, self.d_head).permute(0,2,1,3)
       
    def _attn(self, q, k, v, attn_mask=None):
        scores  = torch.matmul(q, k.transpose(-2, -1))
        if self.scale: scores = scores/math.sqrt(v.size(-1))
        if attn_mask is not None:
            scores = scores.float().masked_fill(attn_mask, -float('inf')).type_as(scores)
        attn_prob  = self.atn_drop(self.softmax(scores))
        attn_vec   = attn_prob @ v
        return attn_vec
   
    def merge_heads(self, x, bs, seq_len):
        x         = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bs, seq_len, -1)
       
    def forward(self, q, k, v, mask=None):
        bs, seq_len = q.size(0), q.size(1)
        wq, wk, wv  = map(lambda o:self.split_heads(*o, bs),
                        zip((q,k,v), (self.wq, self.wk, self.wv)))
        attn_vec    = self._attn(wq, wk, wv)
        attn_vec    = self.merge_heads(attn_vec, bs, seq_len)
        return attn_vec
# %%
class TransformerBlock(nn.Module):
    def __init__(self, d_model=768, n_head=12, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn        = Attention(d_model=768, n_head=12, d_head=64, n_ctx=1024, bias=True, scale=False)
        self.feedforward = FeedForward(dropout=0.1, d_model=768, nx=768*4)
        self.ln_1        = LayerNorm(d_model)
        self.ln_2        = LayerNorm(d_model)
                
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.feedforward(self.ln_2(x))
        return x
# %%
def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for i in range(n)])

class GPT2(nn.Module):
    def __init__(self, nlayers=12, n_ctx=1024, d_model=768, vcb_sz=50257):
        super(GPT2, self).__init__()
        self.nlayers = nlayers
        block        = TransformerBlock(d_model=768, n_head=12, dropout=0.1)
        self.h       = _get_clones(block, 12)
        self.wte     = nn.Embedding(vcb_sz, d_model)
        self.wpe     = nn.Embedding(n_ctx, d_model)
        self.drop    = nn.Dropout(0.1)
        self.ln_f    = LayerNorm(d_model)
        self.out     = nn.Linear(d_model, vcb_sz, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_weights()
    
    def init_weights(self):
        self.out.weight = self.wte.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, src, labels=None, pos_ids=None):
        if pos_ids is None: pos_ids = torch.arange(0, src.size(-1)).unsqueeze(0)
        inp = self.drop((self.wte(src)+self.wpe(pos_ids)))
        for i in range(self.nlayers): inp = self.h[i](inp)
        inp     = self.ln_f(inp)
        logits  = self.out(inp)
        outputs = (logits,) + (inp,)
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs
            return outputs
        return logits
# %%
d_model = 768
n_ctx=1024
x = torch.randint(1, 10000, (1, n_ctx))
# %%
gpt2 = GPT2(vcb_sz=10000)
# %%
output= gpt2(x, x)
# %%
output.shape
# %%
