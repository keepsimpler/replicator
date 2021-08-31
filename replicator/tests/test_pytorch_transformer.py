import torch

from replicator.models.pytorch_transformer import PositionalEncoding, TransformerModel, generate_square_subsequent_mask

def test_positional_encoding():
    batch_size = 2
    seq_len = 32
    embedding_size = 16
    x = torch.rand(batch_size, seq_len, embedding_size)
    positional_encoding = PositionalEncoding(embedding_size=embedding_size)
    output = positional_encoding(x)
    assert tuple(output.shape) == (batch_size, seq_len, embedding_size)

def test_transformer_model():
    vocab_size = 1000 # size of vocabulary
    embedding_size = 200  # embedding dimension
    d_hid = 200 # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability

    seq_len = 100
    transformer_model = TransformerModel(seq_len, vocab_size, embedding_size, nhead, d_hid, nlayers, dropout)

    batch_size = 2
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    x_mask = generate_square_subsequent_mask(seq_len)
    output = transformer_model(x)
    assert tuple(output.shape) == (batch_size, seq_len, vocab_size)

def test_generate_square_subsequent_mask():
    seq_len = 6
    x_mask = generate_square_subsequent_mask(seq_len)
    print(x_mask)
    assert torch.all(torch.abs(torch.exp(x_mask) - torch.tril(torch.ones_like(x_mask), diagonal=0)) < 1e-5)