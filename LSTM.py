import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, hidden_size=512, num_layers=1, padding_idx=None, pooling='last'):
        super(LSTM, self).__init__()
        #########################################  Your Code  ###########################################
        # todo
        # implement lstm
        # the output shape should be batch * 5
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        if pooling == 'max': # in ['max', 'mean', 'sum']:
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'sum':
            self.pooling = lambda x: torch.sum(x, -1)
        elif pooling == 'last':
            self.pooling = lambda x: x[..., -1]
        elif pooling == 'first':
            self.pooling = lambda x: x[..., 0]
        else:
            raise NotImplementedError
        self.fc = nn.Linear(hidden_size * 2, 5)
        # raise NotImplementedError
        #################################################################################################

    def forward(self, inputs, last_hidden=None):
        #inputs  Batch * seq_length

        #########################################  Your Code  ###########################################
        # todo
        # implement lstm
        # the output logits shape should be batch * 5
        emb = self.embedding(inputs)
        if last_hidden is not None:
            outputs, (hidden, cell) = self.lstm(emb, last_hidden)
        else:
            outputs, (hidden, cell) = self.lstm(emb)
        outputs = self.dropout(outputs)
        outputs = self.pooling(outputs.permute(0, 2, 1)).squeeze(-1)
        outputs = self.fc(outputs)
        # raise NotImplementedError
        #################################################################################################
        return outputs





