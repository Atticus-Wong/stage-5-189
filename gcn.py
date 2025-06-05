import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from Dataset_Loader_Node_Classification import Dataset_Loader
import math
import matplotlib.pyplot as plt

# GCN Model Definition
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # Add input dropout for better regularization
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def main():
    # Load data
    citeseer_dataset = Dataset_Loader(dName='citeseer')
    citeseer_dataset.dataset_source_folder_path = 'stage_5_data/citeseer/'
    data = citeseer_dataset.load()

    adj = data['graph']['utility']['A']
    features = data['graph']['X']
    labels = data['graph']['y']
    idx_train = data['train_test_val']['idx_train']
    idx_val = data['train_test_val']['idx_val']
    idx_test = data['train_test_val']['idx_test']

    # Model and Optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=8,
                nclass=labels.max().item() + 1,
                dropout=0.5)

    optimizer = optim.Adam(model.parameters(),
                           lr=0.005, weight_decay=5e-4)

    # Training
    def train(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        print(f'Epoch: {epoch:04d}',
              f'loss_train: {loss_train.item():.4f}',
              f'acc_train: {acc_train.item():.4f}',
              f'loss_val: {loss_val.item():.4f}',
              f'acc_val: {acc_val.item():.4f}')
        
        return loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(1, 301):
        tr_loss, tr_acc, v_loss, v_acc = train(epoch)
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
    
    # Testing
    def test():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\nTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    test()

    # Plotting Learning Curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, 301), train_losses, label='Train Loss')
    plt.plot(range(1, 301), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 301), train_accs, label='Train Accuracy')
    plt.plot(range(1, 301), val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main() 