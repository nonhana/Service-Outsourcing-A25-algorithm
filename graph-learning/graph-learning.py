import time
import os
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 画点函数
def visualize_embedding(h, color, epoch=None, loss=None):
    # figsize:生成图像大小
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch:{epoch},Loss:{loss.item():.4f}', fontsize=16)
    plt.show()


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(520)
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv1 = GCNConv(self.num_features, 16)
        self.conv2 = GCNConv(16, self.num_classes)
        self.classifier = Linear(self.num_classes, self.num_classes)

    def forward(self, x, edge_index):
        # 2层GCN
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        # 分类层
        out = self.classifier(h)
        return out, h


def get_val_loss(model, loader):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    val_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        loss = loss_function(out[data.val_mask], data.y[data.val_mask])
        val_loss += loss.item() * data.num_graphs
    val_loss /= len(loader.dataset)
    model.train()
    return val_loss


# 训练函数
def train(model, loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, h = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.num_graphs
    train_loss /= len(loader.dataset)
    return train_loss, h


def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        _, pred = model(data.x, data.edge_index).max(dim=1)
        correct += int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / len(loader.dataset)
    print('GCN Accuracy: {:.4f}'.format(acc))
    model.train()
    return acc


if __name__ == '__main__':
    # 不加这个可能会报错
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 数据集准备
    dataset = TUDataset
