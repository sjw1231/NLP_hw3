import torch
import torch.nn as nn

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, padding_idx=None):
        super(FastText, self).__init__()
        #########################################  Your Code  ###########################################
        # todo
        # implement fast text
        # the output shape should be batch * 5
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.linear = nn.Linear(embedding_size, 5)
        # raise NotImplementedError
        #################################################################################################

    def forward(self, inputs):
        #inputs  Batch * seq_length

        #########################################  Your Code  ###########################################
        # todo
        # implement fast text
        # the output logits shape should be batch * 5
        outputs = self.embedding(inputs) # Batch * seq_length * embedding_size
        outputs = torch.mean(outputs, dim=1) # Batch * embedding_size
        outputs = self.linear(outputs) # Batch * 5
        # raise NotImplementedError
        #################################################################################################
        return outputs





