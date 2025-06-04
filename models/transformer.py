import torch
import torch.nn as nn
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)  # (1, d_model, max_len)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, channels, time)
        x = x + self.pe[:, :, :x.size(2)]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Conv1d(input_dim, embed_dim, kernel_size=1)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = embed_dim

    def forward(self, x):
        # x shape: (batch, channels, time)
        x = self.embedding(x)  # (B, embed_dim, T)
        x = self.pos_encoder(x)  # (B, embed_dim, T)
        x = rearrange(x, 'b c t -> b t c')  # transformer expects (B, T, C)
        x = self.transformer_encoder(x)  # (B, T, C)
        x = x.mean(dim=1)  # global average pooling
        return x
