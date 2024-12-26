import torch
from torch import nn
from layer.Transformer_EncDec import Encoder, EncoderLayer
from layer.SelfAttention_Family import FullAttention, AttentionLayer
from layer.Embed import PatchEmbedding

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class TaskFusion(nn.Module):
    def __init__(self, configs):
        super().__init__()

        sequence_len = configs.sequence_len
        self.dropout = nn.Dropout(configs.dropout)
        self.value_embedding = nn.Linear(sequence_len, configs.d_model, bias=False)
        self.flatten = nn.Flatten(start_dim=-2)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

    def forward(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.value_embedding(x_enc)
        enc_out = self.dropout(enc_out)
        # [bs x nvars x d_model]

        enc_out, attns = self.encoder(enc_out)

        return enc_out, attns

