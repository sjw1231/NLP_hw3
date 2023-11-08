import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, hidden_size=512, padding_idx=None):
        super(CNN, self).__init__()
        #########################################  Your Code  ###########################################
        # todo
        # implement text cnn
        # the output shape should be batch * 5
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.relu = nn.ReLU()
        self.conv = nn.ModuleList([nn.Conv1d(embedding_size, hidden_size, kernel_size=kernel_size) for kernel_size in [3, 4, 5]])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 3, 5)
        # raise NotImplementedError
        #################################################################################################

    def forward(self, inputs):
        #inputs  Batch * seq_length

        #########################################  Your Code  ###########################################
        # todo
        # implement text cnn
        # the output logits shape should be batch * 5
        emb = self.embedding(inputs) # Batch * seq_length * embedding_size
        emb = emb.permute(0, 2, 1) # Batch * embedding_size * seq_length
        conv_out = [self.relu(conv(emb)) for conv in self.conv] # Batch * hidden_size * (seq_length - kernel_size + 1)
        pool_out = [self.pool(out).squeeze(-1) for out in conv_out] # Batch * hidden_size
        outputs = self.dropout(torch.cat(pool_out, dim=-1)) # Batch * (hidden_size * len(kernel_size))
        outputs = self.fc(outputs) # Batch * 5
        # raise NotImplementedError
        #################################################################################################
        return outputs





