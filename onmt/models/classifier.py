import torch
import torch.nn.functional as F

from onmt.utils.misc import sequence_mask


class Classifier(torch.nn.Module):

    def __init__(self, input_dim, n_class, dropout=None):
        super().__init__()
        self.input_dim = input_dim
        self.n_class = n_class

        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.dropout = (torch.nn.Dropout(dropout) 
                        if dropout is not None else None)
        self.fc2 = torch.nn.Linear(input_dim, n_class)

    def forward(self, memory_bank, lengths): #encoder_out
        # memory_bank: [maxlen, B, H]
        # lengths: [B, ]
        mask = sequence_mask(lengths).float() # [B, maxlen]
        mask = mask / lengths.unsqueeze(1).float() # [B, maxlen]
        # arg1: [B, 1, maxlen], arg2: [B, maxlen, H]] ==> [B, H]
        mean = torch.bmm(mask.unsqueeze(1), 
                         memory_bank.transpose(0, 1)).squeeze(1)

        x = torch.tanh(self.fc1(mean))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1, dtype=torch.float32).type_as(x)

    def update_dropout(self, dropout):
        self.dropout.p = dropout


class Classifier2(torch.nn.Module):

    def __init__(self, input_dim, n_class, dropout=None):
        super().__init__()
        self.input_dim = input_dim
        self.n_class = n_class

        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.dropout = (torch.nn.Dropout(dropout) 
                        if dropout is not None else None)
        self.fc2 = torch.nn.Linear(input_dim, input_dim)
        self.dropout2 = (torch.nn.Dropout(dropout) 
                        if dropout is not None else None)
        self.fc3 = torch.nn.Linear(input_dim, n_class)

    def forward(self, memory_bank, lengths): #encoder_out
        # memory_bank: [maxlen, B, H]
        # lengths: [B, ]
        mask = sequence_mask(lengths).float() # [B, maxlen]
        mask = mask / lengths.unsqueeze(1).float() # [B, maxlen]
        # arg1: [B, 1, maxlen], arg2: [B, maxlen, H]] ==> [B, H]
        mean = torch.bmm(mask.unsqueeze(1), 
                         memory_bank.transpose(0, 1)).squeeze(1)

        x = torch.tanh(self.fc1(mean))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout2 is not None:
            x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1, dtype=torch.float32).type_as(x)

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.dropout2.p = dropout