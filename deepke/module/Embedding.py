import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):
    def __init__(self, config):
        """
        word embedding: 一般 0 为 padding
        pos embedding:  一般 0 为 padding
        dim_strategy: [cat, sum]  多个 embedding 是拼接还是相加
        """
        super(Embedding, self).__init__()

        # self.xxx = config.xxx
        self.vocab_size = config.vocab_size
        self.word_dim = config.word_dim
        self.pos_size = config.pos_size
        self.pos_dim = config.pos_dim if config.dim_strategy == 'cat' else config.word_dim
        self.dim_strategy = config.dim_strategy

        self.wordEmbed = nn.Embedding(self.vocab_size,self.word_dim,padding_idx=0)
        self.posEmbed = nn.Embedding(self.pos_size,self.pos_dim,padding_idx=0)

    def forward(self, x):
        word, pos = x
        word_embedding = self.wordEmbed(word)
        pos_embedding = self.posEmbed(pos)

        if self.dim_strategy == 'cat':
            return torch.cat((word_embedding,pos_embedding), -1)
        elif self.dim_strategy == 'sum':
            # 此时 pos_dim == word_dim
            return word_embedding + pos_embedding
        else:
            raise Exception('dim_strategy must choose from [sum, cat]')