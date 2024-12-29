import torch
from torch import nn
from layer.Transformer_EncDec import Encoder, EncoderLayer
from layer.SelfAttention_Family import FullAttention, AttentionLayer
from layer.Embed import PositionalEmbedding

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
        self.value_embedding = nn.Linear(configs.patch_linear_size, configs.d_model, bias=False)
        self.ln = nn.LayerNorm(configs.sequence_len)
        self.flatten = nn.Flatten(start_dim=-2)

        self.position_embedding = PositionalEmbedding(configs.d_model)

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
        # x_enc = x_enc.permute(0, 2, 1)
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc = x_enc / stdev
        # x_enc = x_enc.permute(0, 2, 1)
        input = x_enc
        n_vars = x_enc.shape[1]
        n_batches = x_enc.shape[0]
        x_enc = x_enc.reshape(x_enc.shape[0] * n_vars, -1, 1)
        enc_out = self.value_embedding(x_enc) + self.position_embedding(x_enc)
        # enc_out = x_enc
        # enc_out = self.dropout(enc_out)
        # [bs x nvars x d_model]

        enc_out, attns = self.encoder(enc_out)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.mean(dim=-1)
        enc_out = self.ln(enc_out)
        attn_last_attn = attns[-1].mean(dim=-1).mean(dim=1)
        attn = torch.reshape(attn_last_attn, (n_batches, n_vars, -1))
        attn = self.ln(input.mul(attn))
        return enc_out, attn

