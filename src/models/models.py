import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers import *
from src.loss import *


class InterfaceSequenceTransformerEncoder(nn.Module):

    def __init__(self,
                 apiid_hidden_size,
                 apiid_vocab_size,
                 interval_hidden_size,
                 interval_vocab_size,
                 attention_hidden_size,
                 num_blocks,
                 num_heads,
                 attention_ff_dim,
                 sequence_length,
                 layer_norm_eps=1e-5,
                 dropout=0.2
    ):
        super(InterfaceSequenceTransformerEncoder, self).__init__()
        self.embedding = ConcatEmbedding(apiid_hidden_size,
                                         apiid_vocab_size,
                                         interval_hidden_size,
                                         interval_vocab_size,
                                         attention_hidden_size,
                                         layer_norm_eps,
                                         dropout
                                         )
        self.position_embedding = LearnablePositionEmbedding(sequence_length, attention_hidden_size)
        self.segment_embedding = nn.Embedding(sequence_length, attention_hidden_size)
        self.transformer_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            self.transformer_blocks.append(TransformerBlock(attention_hidden_size, num_heads, attention_ff_dim, dropout)) # (seq, batch, feature)

    def forward(self, apiid, interval, segment, mask):
        x = self.embedding(apiid, interval)
        x = x.permute(1, 0, 2)
        segment = segment.permute(1, 0)
        x = x + self.position_embedding(x) + self.segment_embedding(segment)
        for idx, block in enumerate(self.transformer_blocks): # (seq, batch, embed_dim)
            x = block(x, mask)
        return x  # (seq, batch, feature)


class InterfaceSequenceTransformerModel(nn.Module):

    def __init__(self,
                 apiid_hidden_size,
                 apiid_vocab_size,
                 interval_hidden_size,
                 interval_vocab_size,
                 attention_hidden_size,
                 num_blocks,
                 num_heads,
                 attention_ff_dim,
                 output_ff_dim,
                 sequence_length,
                 epsilon=1,
                 layer_norm_eps=1e-5,
                 dropout=0.2
    ):
        super(InterfaceSequenceTransformerModel, self).__init__()
        self.encoder = InterfaceSequenceTransformerEncoder(
             apiid_hidden_size,
             apiid_vocab_size,
             interval_hidden_size,
             interval_vocab_size,
             attention_hidden_size,
             num_blocks,
             num_heads,
             attention_ff_dim,
             sequence_length,
             layer_norm_eps,
             dropout
        )

        self.ApiidDistributionOutputLayer = DiscreteDistributionOutputLayer(attention_hidden_size, output_ff_dim, apiid_vocab_size-2) # 去掉mask和cls特殊字段
        self.IntervalDistributionOutputLayer = DiscreteDistributionOutputLayer(attention_hidden_size, output_ff_dim, interval_vocab_size-2)
        self.apiid_reconstruction_loss = ApiidReconstructionKLDivergenceLoss()
        self.interval_reconstruction_loss = IntervalReconstructionWassersteinLoss()
        self.epsilon = epsilon

    def forward(self,
                input_apiid_sequence,
                input_interval_sequence,
                input_segment,
                attention_masks=None,
                label_apiid_dist=None,
                label_interval_dist=None
                ):
        hidden_state = self.encoder(input_apiid_sequence, input_interval_sequence, input_segment, attention_masks)
        cls_hidden_state = hidden_state[0, :, :] # (batch, feature)
        apiid_reconstruction_loss = None
        interval_reconstruction_loss = None
        loss = None

        if label_apiid_dist is not None and label_interval_dist is not None:
            apiid_output = self.ApiidDistributionOutputLayer(cls_hidden_state)
            apiid_reconstruction_loss = self.apiid_reconstruction_loss(apiid_output, label_apiid_dist)
            interval_output = self.IntervalDistributionOutputLayer(cls_hidden_state)
            interval_reconstruction_loss = self.interval_reconstruction_loss(interval_output, label_interval_dist)
            loss = apiid_reconstruction_loss + self.epsilon*interval_reconstruction_loss

        return {
            'cls_hidden_state': cls_hidden_state,
            'loss':loss,
            'apiid_reconstruction_loss': apiid_reconstruction_loss,
            'interval_reconstruction_loss': interval_reconstruction_loss
        }


