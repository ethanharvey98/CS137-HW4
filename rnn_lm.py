# You are supposed to put your own RNN model here

import torch.nn as nn

class SentenceModel(nn.Module):
    def __init__(self, freq):
        super(SentenceModel, self).__init__()
        self.emb = nn.Embedding(num_embeddings=freq.shape[0]+3, embedding_dim=256)
        self.rnn = nn.RNN(input_size=256, hidden_size=1000, num_layers=1, batch_first=True)
        self.linear= nn.Linear(in_features=1000, out_features=freq.shape[0]+3)
        
    def forward(self, x):
        h = self.emb(x)
        h, _ = self.rnn(h)
        out = self.linear(h)
        return out