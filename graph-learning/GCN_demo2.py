import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(520)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def ttt():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


if __name__ == '__main__':
    # 数据准备
    dataset = Planetoid(root='../data/Planetoid', name='Cora',
                        transform=NormalizeFeatures())  # transform预处理
    data = dataset[0]
    # 模型建立
    model = GCN(hidden_channels=16)
    print(model)
    print("=" * 50)
    criterion = torch.nn.CrossEntropyLoss()
    # Define optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)

    # 迭代训练
    for epoch in range(1, 201):
        loss = train()
        if epoch % 20 == 0:
            print(f'Epoch:{epoch} , loss:{loss.item()}')

    # 测试
    print("正确率：", ttt())
