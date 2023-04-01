import os
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.nn import Linear
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataSet:
    def __init__(self):
        # 定义节点特征向量x和标签向量y
        x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
        # 定义边矩阵
        edge_index = torch.tensor([[0, 1, 2, 0, 3],  # 起始点
                                   [1, 0, 1, 3, 2]], dtype=torch.long)  # 终止点
        # 定义train_mask(二进制掩码，为True则代表该数据作为有效数据参加训练)
        train_mask = torch.tensor([True, True, True, True])
        # 构建torch_geometric.data.Data
        self.data = Data(x=x, y=y, edge_index=edge_index,
                         train_mask=train_mask)
        # 手动引入特征向量的种类个数num_classes
        self.data.num_classes = int(torch.max(self.data.y).item() + 1)


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(520)  # 固定随机数种子，保证结果的可重复性
        self.num_features = num_features  # 节点特征的维度
        self.num_classes = num_classes  # 节点分类的类别数
        # 第一层GAT卷积层
        self.conv1 = GATConv(self.num_features, 16, heads=4, concat=True)
        # 第二层GAT卷积层
        self.conv2 = GATConv(16 * 4, self.num_classes, heads=1, concat=True)
        # 线性分类器层
        self.classifier = Linear(self.num_classes, self.num_classes)

    def forward(self, x, edge_index):
        # 通过第一层GAT卷积层，得到节点的特征表示
        h = self.conv1(x, edge_index)
        # 对特征表示进行ReLU激活函数处理
        h = h.relu()
        # 通过第二层GAT卷积层，得到节点的特征表示
        h = self.conv2(h, edge_index)
        # 对特征表示进行ReLU激活函数处理
        h = h.relu()
        # 使用线性分类器层对节点特征进行映射，得到节点的分类概率
        out = self.classifier(h)
        # 返回节点分类结果和中间层特征表示
        return out, h


def get_val_loss(model, data):
    # 切换模型为评估模式
    model.eval()
    # 使用模型进行预测，获取输出
    out = model(data)
    # 定义损失函数，这里使用交叉熵损失函数
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    # 计算验证集上的损失
    # 首先选取数据集中属于验证集的部分进行计算，即data.val_mask为True的部分
    # 然后选取预测结果中与验证集对应的部分，即out[data.val_mask]
    # 最后计算预测结果和真实标签之间的交叉熵损失
    loss = loss_function(out[data.val_mask], data.y[data.val_mask])
    # 切换回模型为训练模式
    model.train()
    # 返回验证集上的损失值
    return loss.item()


# 训练函数
def train(data):
    # 梯度清零
    optimizer.zero_grad()
    # 使用模型进行预测，获取输出和中间特征向量
    out, h = model(data.x, data.edge_index)
    # 计算损失值，这里使用交叉熵损失函数
    # 首先选取数据集中属于训练集的部分进行计算，即data.train_mask为True的部分
    # 然后选取预测结果中与训练集对应的部分，即out[data.train_mask]
    # 最后计算预测结果和真实标签之间的交叉熵损失
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    # 反向传播，计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()
    # 返回损失值和中间特征向量h
    return loss, h


# 测试函数
def test(model, data):
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        # 将out的形状从(batch_size, num_classes * heads)改为(batch_size, num_classes)
        out = out.view(-1, dataset.num_classes)
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
    dataset = DataSet().data
    # 声明GCN模型
    model = GCN(dataset.num_features, dataset.num_classes)
    # 损失函数 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器 Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 训练，迭代重复1001次
    for epoch in range(1001):
        loss, h = train(dataset)
    # 测试
    test_acc = test(model=model, data=dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
