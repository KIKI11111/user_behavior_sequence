import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ConcatEmbedding(nn.Module):

    def __init__(self,
                 apiid_hidden_size,
                 apiid_vocab_size,
                 interval_hidden_size,
                 interval_vocab_size,
                 out_hidden_size,
                 layer_norm_eps,
                 dropout
    ):
        super(ConcatEmbedding, self).__init__()
        self.apiid_embedding = nn.Embedding(apiid_vocab_size, apiid_hidden_size)
        self.interval_embedding = nn.Embedding(interval_vocab_size, interval_hidden_size)
        self.fc = nn.Linear(apiid_hidden_size+interval_hidden_size, out_hidden_size)
        self.LayerNorm = nn.LayerNorm(out_hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, apiid, interval):
        out = torch.cat((self.apiid_embedding(apiid), self.interval_embedding(interval)), dim=2)
        out = F.relu(out)
        out = self.fc(out)
        out = F.relu(out)
        out = self.LayerNorm(out)
        return self.dropout(out)


class AddEmbedding(nn.Module):

    def __init__(self,
                 apiid_vocab_size,
                 interval_vocab_size,
                 out_hidden_size,
                 layer_norm_eps,
                 dropout
    ):
        super(AddEmbedding, self).__init__()
        self.apiid_embedding = nn.Embedding(apiid_vocab_size, out_hidden_size)
        self.interval_embedding = nn.Embedding(interval_vocab_size, out_hidden_size)
        self.LayerNorm = nn.LayerNorm(out_hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, apiid, interval):

        out = self.apiid_embedding(apiid) + self.interval_embedding(interval)
        out = F.relu(out)
        out = self.LayerNorm(out)
        return self.dropout(out)


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.LayerNorm1 = nn.LayerNorm(embed_dim)
        self.LayerNorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, x, mask=None): # (seq, batch, feature)
        sequence_length, batch_size, feature = x.shape
        num_heads = self.num_heads

        if mask is not None: # (batch, seq)
            mask = mask.unsqueeze(1).repeat(1, sequence_length, 1) # (batch, seq, seq)
            mask = mask.unsqueeze(0).repeat(num_heads, 1, 1, 1).view(batch_size*num_heads, sequence_length, sequence_length) # (batch*num_heads, seq, seq)

        attn_output = self.attention(x, x, x, attn_mask=mask)
        x = self.LayerNorm1(x*self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.LayerNorm2(x*self.dropout2(ffn_output))
        return x


class LearnablePositionEmbedding(nn.Module):

    def __init__(self, max_len, hidden_size):
        super(LearnablePositionEmbedding, self).__init__()
        # 可学习的位置嵌入
        self.position_embedding = nn.Embedding(max_len, hidden_size)

    def forward(self, x): # (seq, batch, feature)
        s, b, f = x.shape
        positions = torch.arange(s, device=x.device).unsqueeze(1).repeat(1, b) # (seq, batch)
        # 获取位置嵌入
        return self.position_embedding(positions)


class DiscreteDistributionOutputLayer(nn.Module):
    # 输入序列表征， 输出apiid的离散分布的预测向量
    def __init__(self, attention_hidden_size, ff_size, out_size):
        super(DiscreteDistributionOutputLayer, self).__init__()
        self.fc1 = nn.Linear(attention_hidden_size, ff_size)
        self.fc2 = nn.Linear(ff_size, out_size)

    def forward(self, x): # 取cls 或者pooling的结果，(batch, feature)
        out = self.fc1(x)
        out = F.relu(out)
        return self.fc2(out)


class NormalDistributionOutputLayer(nn.Module):
    # 输入序列表征，
    def __init__(self, attention_hidden_size, ff_size):
        super(NormalDistributionOutputLayer, self).__init__()
        self.fc1 = nn.Linear(attention_hidden_size, ff_size)
        self.fc_mu = nn.Linear(attention_hidden_size, 1)
        self.fc_sigma = nn.Linear(attention_hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        return torch.cat([self.fc_mu(out), self.fc_sigma(out)], dim=1)

