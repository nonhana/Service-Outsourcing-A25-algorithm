import os
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import torch
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv


# 实现邻接表
class Vertex:
    def __init__(self, key, type, name):
        self.id = key
        self.type = type
        self.name = name
        self.connectedTo = {}

    # 从这个顶点添加一个连接到另一个
    def addNeighbor(self, nbr, name, weight=0):
        self.connectedTo[nbr] = [weight, name]

    # 修改str
    def __str__(self):
        return str(self.id) + 'connectedTo' + str(
            [x.id for x in self.connectedTo])

    # 返回邻接表中的所有的项点
    def getConnections(self):
        return self.connectedTo.items()

    def getId(self):
        return self.id

    # 返回从这个顶点到作为参数顶点的边的权重和名字
    def getweight(self, nbr):
        return self.connectedTo[nbr]


# 实现图
class Graph:
    def __init__(self):
        # {}代表字典
        self.vertList = {}
        self.matrix = []
        self.numVertices = 0
        self.visble = nx.Graph()
        self.feature_vector = []
        self.labels = []
        self.edge_matrix = []

    # 增加顶点，具有属性：id，类型，名称
    def addVertex(self, key, type, name):
        self.labels.append(self.numVertices)
        self.numVertices += 1
        self.visble.add_node(key)
        newVertex = Vertex(key, type, name)
        self.vertList[key] = newVertex
        return newVertex

    # 返回某个顶点的信息
    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    # 判断顶点是否在邻接表中
    def __contains__(self, n):
        return n in self.vertList

    # 增加边
    def addEdge(self, f, t, name, const=0):
        # 起始点，目标点，权重。
        # 注意点：f,t是vertlist中的数组下标，不是target的id
        if f not in self.vertList:
            nv = self.addVertex(f, "default_type", "default_name")
        if t not in self.vertList:
            nv = self.addVertex(t, "default_type", "default_name")
        self.matrix[f][t] = const
        self.visble.add_edge(f, t)
        self.vertList[f].addNeighbor(self.vertList[t], name, const)

    # 初始化邻接矩阵
    def initMatrix(self, nodenum):
        for i in range(nodenum):
            row = []
            for j in range(nodenum):
                row.append(0)
            self.matrix.append(row)

    # 遍历邻接矩阵
    def printMatrix(self):
        for i in range(len(self.matrix)):
            print(self.matrix[i])

    # 计算邻接矩阵的特征值、特征向量
    def coculate(self):
        mat = np.array(self.matrix)
        eigenvalue, featurevector = np.linalg.eig(mat)
        # print("特征值：", eigenvalue)
        # print("特征向量：", featurevector)
        # 将每个点的特征向量从计算结果中取出
        for i in range(len(featurevector)):
            feature_list = []
            for j in range(len(featurevector)):
                feature_list.append(featurevector[j][i])
            self.feature_vector.append(feature_list)
        return

    # 根据邻接矩阵写出边矩阵
    def build_edge_matrix(self):
        start = []
        end = []
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if self.matrix[j][i] != 0:
                    start.append(j)
                    end.append(i)
        self.edge_matrix.append(start)
        self.edge_matrix.append(end)
        return

    # 获取所有顶点
    def getVertices(self):
        length = 0
        for item in self.vertList.values():
            # print("("+"类型："+item.type+"，"+"名称："+item.name+")")
            length = length+1
        return length

    # 使用迭代器返回所有的邻接表信息
    def __iter__(self):
        return iter(self.vertList.values())


# 构建GCN神经网络
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(520)
        self.num_features = num_features
        self.num_classes = num_classes
        self.conv1 = GCNConv(self.num_features, 4)  # 只定义子输入特证和输出特证即可
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, self.num_classes)

    def forward(self, x, edge_index):
        # 3层GCN
        h = self.conv1(x, edge_index)  # 给入特征与邻接矩阵（注意格式，上面那种）
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        # 分类层
        out = self.classifier(h)
        return out, h


# 画点函数
def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch:{epoch},Loss:{loss.item():.4f}', fontsize=16)
    plt.show()


# 训练函数
def train(data):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h


if __name__ == "__main__":
    # 添加顶点
    g = Graph()
    # 根节点
    g.addVertex(0, "industry", "基础化工")
    # 一级产业
    g.addVertex(1, "industry", "非金属材料II")
    g.addVertex(2, "industry", "橡胶")
    g.addVertex(3, "industry", "塑料")
    g.addVertex(4, "industry", "化学纤维")
    g.addVertex(5, "industry", "化学制品")
    g.addVertex(6, "industry", "化学原料")
    g.addVertex(7, "industry", "农化制品")
    # 二级产业
    g.addVertex(8, "industry", "涤纶")
    g.addVertex(9, "industry", "粘胶")
    g.addVertex(10, "industry", "锦纶")
    g.addVertex(11, "industry", "其他化学纤维")
    g.addVertex(12, "industry", "氨纶")
    # 公司
    g.addVertex(13, "company", "海利得")
    g.addVertex(14, "company", "苏州龙杰")
    g.addVertex(15, "company", "优彩资源")
    g.addVertex(16, "company", "华西股份")
    g.addVertex(17, "company", "新凤鸣")
    # 产品
    g.addVertex(g.getVertices(), "product", "土工格栅材料")
    g.addVertex(g.getVertices(), "product", "灯箱广告材料")
    g.addVertex(g.getVertices(), "product", "PVC涂层材料")
    g.addVertex(g.getVertices(), "product", "聚酯工业长丝")
    g.addVertex(g.getVertices(), "product", "装饰材料")
    g.addVertex(g.getVertices(), "product", "石塑地板")
    g.addVertex(g.getVertices(), "product", "装饰膜")
    g.addVertex(g.getVertices(), "product", "灯箱布")
    g.addVertex(g.getVertices(), "product", "涤纶工业长丝")
    g.addVertex(g.getVertices(), "product", "PVC膜")
    g.addVertex(g.getVertices(), "product", "轮胎帘子布")
    g.addVertex(g.getVertices(), "product", "电脑喷绘胶片布")
    g.addVertex(g.getVertices(), "product", "篷盖材料")
    g.addVertex(g.getVertices(), "product", "聚酯切片")

    # 初始化邻接矩阵
    g.initMatrix(g.getVertices())

    # 添加边和权重
    # 一级产业-->根节点
    g.addEdge(1, 0, '上级行业', 1)
    g.addEdge(2, 0, '上级行业', 1)
    g.addEdge(3, 0, '上级行业', 1)
    g.addEdge(4, 0, '上级行业', 1)
    g.addEdge(5, 0, '上级行业', 1)
    g.addEdge(6, 0, '上级行业', 1)
    g.addEdge(7, 0, '上级行业', 1)
    # 二级产业-->一级产业
    g.addEdge(8, 4, '上级行业', 1)
    g.addEdge(9, 4, '上级行业', 1)
    g.addEdge(10, 4, '上级行业', 1)
    g.addEdge(11, 4, '上级行业', 1)
    g.addEdge(12, 4, '上级行业', 1)
    # 公司-->二级产业
    g.addEdge(13, 8, '所属行业', 1)
    g.addEdge(14, 8, '所属行业', 1)
    g.addEdge(15, 8, '所属行业', 1)
    g.addEdge(16, 8, '所属行业', 1)
    g.addEdge(17, 8, '所属行业', 1)
    # 产品-->公司
    g.addEdge(18, 13, '主营产品', 0.001)
    g.addEdge(19, 13, '主营产品', 0.20321)
    g.addEdge(20, 13, '主营产品', 0.001)
    g.addEdge(21, 13, '主营产品', 0.001)
    g.addEdge(22, 13, '主营产品', 0.62828)
    g.addEdge(23, 13, '主营产品', 0.041549)
    g.addEdge(24, 13, '主营产品', 0.064964)
    g.addEdge(25, 13, '主营产品', 0.0210146)
    g.addEdge(26, 13, '主营产品', 0.624192)
    g.addEdge(27, 13, '主营产品', 0.051554)
    g.addEdge(28, 13, '主营产品', 0.11071)
    g.addEdge(29, 13, '主营产品', 0.001)
    g.addEdge(30, 13, '主营产品', 0.001)
    g.addEdge(31, 13, '主营产品', 0.006078)
    g.addEdge(31, 17, '主营产品', 0.0011044)

    # 画出拓扑结构
    # nx.draw(g.visble, with_labels=True, node_size=800, node_color="green")
    # plt.show()

    # 计算邻接矩阵的特征值和特征向量
    g.coculate()

    # 定义节点特征向量x和标签y
    x = torch.tensor(g.feature_vector, dtype=torch.float)
    y = torch.tensor(g.labels, dtype=torch.float)

    # 定义边矩阵
    g.build_edge_matrix()
    edge_index = torch.tensor(g.edge_matrix, dtype=torch.long)

    # 定义train_mask
    train_mask = [(True if d is not None else False) for d in y]

    # 构建data
    indusry_data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask)
    print("data:", indusry_data)
    print("train_mask:", indusry_data.train_mask)

    # 不加这个可能会报错
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 数据集准备
    dataset = indusry_data
    # data = dataset[0]

    print(dataset.num_features)

    # # 声明GCN模型
    # model = GCN(dataset.num_features, dataset.num_classes)

    # # 损失函数 交叉熵损失
    # criterion = torch.nn.CrossEntropyLoss()
    # # 优化器 Adam
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # # 训练
    # for epoch in range(401):
    #     loss, h = train(data)
    #     if epoch % 100 == 0:
    #         visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
    #         time.slep(0.3)
