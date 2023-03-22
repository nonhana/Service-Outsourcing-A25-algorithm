import time
import os
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
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


class DataSet:
    def __init__(self):
        # 定义节点特征向量x和标签y
        x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
        # 定义边
        edge_index = torch.tensor([[0, 1, 2, 0, 3],  # 起始点
                                   [1, 0, 1, 3, 2]], dtype=torch.long)  # 终止点
        # 定义train_mask
        train_mask = torch.tensor([True, True, True, True])
        # 构建data
        self.data = Data(x=x, y=y, edge_index=edge_index,
                         train_mask=train_mask)
        self.data.num_classes = int(torch.max(self.data.y).item() + 1)


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


def get_val_loss(model, data):
    model.eval()
    out = model(data)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[data.val_mask], data.y[data.val_mask])
    model.train()
    return loss.item()


# 训练函数
def train(data):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h


# 测试函数
def test(model, data):
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = float(pred[data.train_mask].eq(
            data.y[data.train_mask]).sum().item())
        acc = correct / data.train_mask.sum().item()
    model.train()
    return acc


if __name__ == '__main__':
    # 不加这个可能会报错
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 数据集准备
    data = DataSet()
    dataset = data.data

    # 声明GCN模型
    model = GCN(dataset.num_features, dataset.num_classes)

    # 损失函数 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器 Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练
    for epoch in range(1001):
        loss, h = train(dataset)
        # if epoch % 500 == 0:
        #     visualize_embedding(h, color=dataset.y, epoch=epoch, loss=loss)
        #     time.sleep(0.3)

    # 测试
    test_acc = test(model=model, data=dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
