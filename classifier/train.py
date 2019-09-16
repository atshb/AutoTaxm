import pickle
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data


#
class OntDataset(data.Dataset):
    def __init__(self, dataset, vectorizer):
        self.dataset = dataset
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        a, b, l = self.dataset[idx]
        a = torch.from_numpy(self.vectorizer[a]).float()
        b = torch.from_numpy(self.vectorizer[b]).float()
        l = torch.tensor(l)
        return ((a, b), l)


#
class SimpleConcat(nn.Module):
    def __init__(self):
        super(SimpleConcat, self).__init__()

    def forward(self, x):
        a, b = x
        batch_size = a.size()[0]
        a, b = a.reshape(batch_size, -1), b.reshape(batch_size, -1)
        x = torch.cat([a, b], dim=1)
        return x


# #
# class RnnConcat(nn.Module):
#     def __init__(self, seq_size, h_size):
#         super(RnnConcat, self).__init__()
#         rnn_a = nn.LSTM(seq_size, h_size)
#         rnn_b = nn.LSTM(seq_size, h_size)
#
#     def forward(self, a, b):
#         return torch.concat([rnn_a(a), rnn_b(b)], dim=0)


#
class Ont(nn.Module):
    def __init__(self, concat, x_size, h_size=300, y_size=4, drop_rate=0.5):
        super(Ont, self).__init__()
        self.concat = concat
        self.l1 = nn.Linear(x_size, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, y_size)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.concat(x)
        h = self.drop(F.relu(self.l1(x)))
        h = self.drop(F.relu(self.l2(h)))
        t = self.l3(h)
        return t


def main():
    # param
    train_rate = 0.8
    max_epoch  = 50
    batch_size = 1024

    h_size = 512
    drop_rate = 0.2

    # Loading
    vectorizer    = pd.read_pickle('../vectorizer/vectorizer_w2v.pkl')
    train_dataset = pd.read_pickle('../dataset/wordnet_train.csv')
    valid_dataset = pd.read_pickle('../dataset/wordnet_valid.csv')

    train_dataset = OntDataset(train_dataset, vectorizer)
    valid_dataset = OntDataset(valid_dataset, vectorizer)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Calculate input size from dataset
    max_seq_len, vec_size = vectorizer['example'][0].shape
    x_size = 2 * max_seq_len * vec_size

    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = Ont(SimpleConcat(), x_size=x_size, h_size=h_size, drop_rate=drop_rate).to(device)
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    #
    for epoch in range(max_epoch):
        # Training
        epoch_loss = 0
        model.train()
        for x, y in train_loader:
            t = model(x)
            loss = loss_func(t, y)
            epoch_loss += loss.cpu().item()
            # 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (i + 1) % 10 == 0: print(f'{i + 1:>4} : {loss.cpu().item():>6.3}')

        # Validation
        epoch_accu = 0
        model.eval()
        for x, y in valid_loader:
            t = model(x)
            _, t = torch.max(t.data, 1)
            epoch_accu += sum(1 for t_i, y_i in zip(t, y) if t_i == y_i)


        # Show Progress
        epoch_loss /= n_train
        epoch_accu /= n_valid
        print(f'{epoch:0>2} | loss : {epoch_loss:>7.5f} | accu : {epoch_accu:.2%}')


if __name__ == '__main__': main()
