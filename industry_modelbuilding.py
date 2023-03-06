from py2neo import Graph
import os
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import torch
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv


# 连接Neo4j数据库并读取数据，将数据存到data.txt文件内
class ReadGraph:
    def __init__(self):
        self.g = Graph('http://localhost:7474/', user='neo4j',
                       password='20021209xiang', name='neo4j')
        # 根节点
        self.n_node = []
        # 第一条边的相关属性
        self.r1_startnode = []
        self.r1_name = []
        self.r1_endnode = []
        # 第二节点
        self.m1_node = []
        # 第二条边的相关属性
        self.r2_startnode = []
        self.r2_name = []
        self.r2_endnode = []
        # 第三节点
        self.m2_node = []
        # 第三条边的相关属性
        self.r3_startnode = []
        self.r3_name = []
        self.r3_endnode = []
        # 第四节点==
        self.m3_node = []
        # 第四条边的相关属性
        self.r4_startnode = []
        self.r4_name = []
        self.r4_endnode = []
        # 第五节点(公司)
        self.m4_node = []
        # 第五条边的相关属性
        self.r5_startnode = []
        self.r5_name = []
        self.r5_endnode = []
        # 第六节点(产品小类)
        self.m5_node = []
        # 第六条边的相关属性
        self.r6_startnode = []
        self.r6_name = []
        self.r6_endnode = []
        # 第七节点
        self.m6_node = []

    # 查询

    def query(self):
        # 打开data.txt文件
        file_handle = open('data.txt', mode='w')

        # 定义cql语句
        cql = 'match (n:industry {name:"基础化工"})-[r1]-(m1:industry {name:"塑料"})-[r2]-(m2:industry {name:"膜材料"})-[r3]-(m3)-[r4]-(m4)-[r5]->(m5)-[r6:`上游材料`]-(m6) return n,r1,m1,r2,m2,r3,m3,r4,m4,r5,m5,r6,m6'

        # 查询
        n = self.g.run(cql).data('n')
        r1 = self.g.run(cql).data('r1')
        m1 = self.g.run(cql).data('m1')
        r2 = self.g.run(cql).data('r2')
        m2 = self.g.run(cql).data('m2')
        r3 = self.g.run(cql).data('r3')
        m3 = self.g.run(cql).data('m3')
        r4 = self.g.run(cql).data('r4')
        m4 = self.g.run(cql).data('m4')
        r5 = self.g.run(cql).data('r5')
        m5 = self.g.run(cql).data('m5')
        r6 = self.g.run(cql).data('r6')
        m6 = self.g.run(cql).data('m6')

        # 根节点
        file_handle.write('===根节点start===\n')
        for i in range(len(n)):
            record = list(n[i].values())
            result = list(record[0].values())[1]
            self.n_node.append(result)
            file_handle.write(result+"\n")
        self.n_node = list(set(self.n_node))
        file_handle.write('===根节点end===\n')

        # print(self.n_node)

        # 第一条边
        file_handle.write('===第一条边start===\n')
        for i in range(len(r1)):
            record = list(r1[i].values())
            result = str(record[0])
            # 将字符串切片，取出头结点、边名称、尾节点
            self.r1_startnode.append(
                result[result.index("(")+1:result.index(")")])
            self.r1_name.append(
                result[result.index("[")+2:result.index("]")-3])
            self.r1_endnode.append(
                result[result.index("(", 5)+1:result.index(")", 10)])
        for item in self.r1_startnode:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r1_name:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r1_endnode:
            file_handle.write(item+"\n")
        file_handle.write('===第一条边end===\n')
        # print(self.r1_startnode)
        # print(self.r1_name)
        # print(self.r1_endnode)

        # 第二节点
        file_handle.write('===第二节点start===\n')
        for i in range(len(m1)):
            record = list(m1[i].values())
            result = list(record[0].values())[1]
            self.m1_node.append(result)
            file_handle.write(result+"\n")
        file_handle.write('===第二节点end===\n')
        self.m1_node = list(set(self.m1_node))
        # print(self.m1_node)

        # 第二条边
        file_handle.write('===第二条边start===\n')
        for i in range(len(r2)):
            record = list(r2[i].values())
            result = str(record[0])
            # 将字符串切片，取出头结点、边名称、尾节点
            self.r2_startnode.append(
                result[result.index("(")+1:result.index(")")])
            self.r2_name.append(
                result[result.index("[")+2:result.index("]")-3])
            self.r2_endnode.append(
                result[result.index("(", 5)+1:result.index(")", 10)])
        for item in self.r2_startnode:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r2_name:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r2_endnode:
            file_handle.write(item+"\n")
        file_handle.write('===第二条边end===\n')
        # print(self.r2_startnode)
        # print(self.r2_name)
        # print(self.r2_endnode)

        # 第三节点
        file_handle.write('===第二节点start===\n')
        for i in range(len(m2)):
            record = list(m2[i].values())
            result = list(record[0].values())[1]
            self.m2_node.append(result)
            file_handle.write(result+'\n')
        file_handle.write('===第二节点end===\n')
        self.m2_node = list(set(self.m2_node))
        # print(self.m2_node)

        # 第三条边
        file_handle.write('===第三条边start===\n')
        for i in range(len(r3)):
            record = list(r3[i].values())
            result = str(record[0])
            # 将字符串切片，取出头结点、边名称、尾节点
            self.r3_startnode.append(
                result[result.index("(")+1:result.index(")")])
            self.r3_name.append(
                result[result.index("[")+2:result.index("]")-3])
            self.r3_endnode.append(
                result[result.index("(", 5)+1:result.index(")", 10)])
        for item in self.r3_startnode:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r3_name:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r3_endnode:
            file_handle.write(item+"\n")
        file_handle.write('===第三条边end===\n')
        # print(self.r3_startnode)
        # print(self.r3_name)
        # print(self.r3_endnode)

        # 第四节点
        file_handle.write('===第四节点start===\n')
        for i in range(len(m3)):
            record = list(m3[i].values())
            result = list(record[0].values())[1]
            self.m3_node.append(result)
            file_handle.write(result+'\n')
        file_handle.write('===第四节点end===\n')
        self.m3_node = list(set(self.m3_node))
        # print(self.m3_node)

        # 第四条边
        file_handle.write('===第四条边start===\n')
        for i in range(len(r4)):
            record = list(r4[i].values())
            result = str(record[0])
            # 将字符串切片，取出头结点、边名称、尾节点
            self.r4_startnode.append(
                result[result.index("(")+1:result.index(")")])
            self.r4_name.append(
                result[result.index("[")+2:result.index("]")-3])
            self.r4_endnode.append(
                result[result.index("(", 5)+1:result.index(")", 10)])
        for item in self.r4_startnode:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r4_name:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r4_endnode:
            file_handle.write(item+"\n")
        file_handle.write('===第四条边end===\n')
        # print(self.r4_startnode)
        # print(self.r4_name)
        # print(self.r4_endnode)

        # 第五节点
        file_handle.write('===第五节点start===\n')
        for i in range(len(m4)):
            record = list(m4[i].values())
            result = list(record[0].values())[0]
            self.m4_node.append(result)
            file_handle.write(result+'\n')
        file_handle.write('===第五节点end===\n')
        self.m4_node = list(set(self.m4_node))
        # print(self.m4_node)

        # 第五条边
        file_handle.write('===第五条边start===\n')
        for i in range(len(r5)):
            record = list(r5[i].values())
            result = str(record[0])
            # 将字符串切片，取出头结点、边名称、尾节点
            self.r5_startnode.append(
                result[result.index("(")+1:result.index(")")])
            self.r5_name.append(
                result[result.index("[")+2:result.index("]")-3])
            self.r5_endnode.append(
                result[result.index("(", 5)+1:result.index(")", 10)])
        for item in self.r5_startnode:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r5_name:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r5_endnode:
            file_handle.write(item+"\n")
        file_handle.write('===第五条边end===\n')
        # print(self.r5_startnode)
        # print(self.r5_name)
        # print(self.r5_endnode)

        # 第六节点
        file_handle.write('===第六节点start===\n')
        for i in range(len(m5)):
            record = list(m5[i].values())
            result = list(record[0].values())[0]
            self.m5_node.append(result)
            file_handle.write(result+'\n')
        file_handle.write('===第六节点end===\n')
        self.m5_node = list(set(self.m5_node))
        # print(self.m5_node)

        # 第六条边
        file_handle.write('===第六条边start===\n')
        for i in range(len(r6)):
            record = list(r6[i].values())
            result = str(record[0])
            # 将字符串切片，取出头结点、边名称、尾节点
            self.r6_startnode.append(
                result[result.index("(")+1:result.index("-")-1])
            self.r6_name.append(
                result[result.index("[")+2:result.index("]")-3])
            self.r6_endnode.append(
                result[result.index("]")+4:result.index(")", len(result)-1)])
        for item in self.r6_startnode:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r6_name:
            file_handle.write(item+"\n")
        file_handle.write('\n')
        for item in self.r6_endnode:
            file_handle.write(item+"\n")
        file_handle.write('===第六条边end===\n')
        # print(self.r6_startnode)
        # print(self.r6_name)
        # print(self.r6_endnode)

        # 第七节点
        file_handle.write('===第七节点start===\n')
        for i in range(len(m6)):
            record = list(m6[i].values())
            result = list(record[0].values())[0]
            self.m6_node.append(result)
            file_handle.write(result+'\n')
        file_handle.write('===第七节点end===\n')
        self.m6_node = list(set(self.m6_node))
        # print(self.m6_node)

        file_handle.close()
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
class IndustryGraph:
    def __init__(self):
        # {}代表字典
        self.vertList = {}
        self.matrix = []
        self.numVertices = 0
        self.visble = nx.Graph()
        self.feature_vector = []
        self.labels = []
        self.name_labels = []
        self.edge_matrix = []

    # 增加顶点，具有属性：id，类型，名称
    def addVertex(self, key, type, name):
        self.labels.append(self.numVertices)
        self.name_labels.append(name)
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
        return len(self.vertList.values())

    # 使用迭代器返回所有的邻接表信息
    def __iter__(self):
        return iter(self.vertList.values())


# 构建GCN神经网络
# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(GCN, self).__init__()
#         torch.manual_seed(520)
#         self.num_features = num_features
#         self.num_classes = num_classes
#         self.conv1 = GCNConv(self.num_features, 4)  # 只定义子输入特证和输出特证即可
#         self.conv2 = GCNConv(4, 4)
#         self.conv3 = GCNConv(4, 2)
#         self.classifier = Linear(2, self.num_classes)

#     def forward(self, x, edge_index):
#         # 3层GCN
#         h = self.conv1(x, edge_index)  # 给入特征与邻接矩阵（注意格式，上面那种）
#         h = h.tanh()
#         h = self.conv2(h, edge_index)
#         h = h.tanh()
#         h = self.conv3(h, edge_index)
#         h = h.tanh()
#         # 分类层
#         out = self.classifier(h)
        # return out, h


# 画点函数
# def visualize_embedding(h, color, epoch=None, loss=None):
#     plt.figure(figsize=(7, 7))
#     plt.xticks([])
#     plt.yticks([])
#     h = h.detach().cpu().numpy()
#     plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
#     if epoch is not None and loss is not None:
#         plt.xlabel(f'Epoch:{epoch},Loss:{loss.item():.4f}', fontsize=16)
#     plt.show()


# 训练函数
# def train(data):
#     optimizer.zero_grad()
#     out, h = model(data.x, data.edge_index)
#     loss = criterion(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#     return loss, h

if __name__ == "__main__":
    # 连接数据库并读取数据
    handler = ReadGraph()
    handler.query()

    # 添加顶点
    g = IndustryGraph()
    # 根节点
    g.addVertex(g.getVertices(), "industry", handler.n_node[0])
    # 一级产业
    for item in handler.m1_node:
        g.addVertex(g.getVertices(), "industry", item)
    # 二级产业
    for item in handler.m2_node:
        g.addVertex(g.getVertices(), "industry", item)
    # 公司
    for item in handler.m3_node:
        g.addVertex(g.getVertices(), "company", item)
    # 主营产品
    for item in handler.m4_node:
        g.addVertex(g.getVertices(), "product", item)
    # 产品小类
    for item in handler.m5_node:
        g.addVertex(g.getVertices(), "littleproduct", item)
    # 上游材料
    for item in handler.m6_node:
        g.addVertex(g.getVertices(), "material", item)

    # 初始化邻接矩阵
    g.initMatrix(g.getVertices())

    # 添加边和权重
    # 一级产业-->根节点
    for i in range(len(handler.r1_endnode)):
        g.addEdge(g.name_labels.index(handler.r1_startnode[i]), g.name_labels.index(
            handler.r1_endnode[i]), handler.r1_name[i], 1)
    # 二级产业-->一级产业
    for i in range(len(handler.r2_endnode)):
        g.addEdge(g.name_labels.index(handler.r2_startnode[i]), g.name_labels.index(
            handler.r2_endnode[i]), handler.r2_name[i], 1)
    # 公司-->二级产业
    for i in range(len(handler.r3_endnode)):
        g.addEdge(g.name_labels.index(handler.r3_startnode[i]), g.name_labels.index(
            handler.r3_endnode[i]), handler.r3_name[i], 1)
    # 产品-->公司
    for i in range(len(handler.r4_endnode)):
        g.addEdge(g.name_labels.index(handler.r4_startnode[i]), g.name_labels.index(
            handler.r4_endnode[i]), handler.r4_name[i], 1)
    # 产品小类-->产品
    for i in range(len(handler.r5_endnode)):
        g.addEdge(g.name_labels.index(handler.r5_startnode[i]), g.name_labels.index(
            handler.r5_endnode[i]), handler.r5_name[i], 1)
    # 上游材料-->产品小类
    for i in range(len(handler.r6_endnode)):
        g.addEdge(g.name_labels.index(handler.r6_startnode[i]), g.name_labels.index(
            handler.r6_endnode[i]), handler.r6_name[i], 1)

    # 输出邻接矩阵
    # print(g.matrix)

    # 画出拓扑结构
    nx.draw(g.visble, node_size=100, node_color="skyblue")
    plt.show()

    # 计算邻接矩阵的特征值和特征向量
    # g.coculate()

    # 定义节点特征向量x和标签y
    # x = torch.tensor(g.feature_vector, dtype=torch.float)
    # y = torch.tensor(g.labels, dtype=torch.float)

    # 定义边矩阵
    # g.build_edge_matrix()
    # edge_index = torch.tensor(g.edge_matrix, dtype=torch.long)

    # 定义train_mask
    # train_mask = [(True if d is not None else False) for d in y]

    # 构建data
    # indusry_data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask)
    # print("data:", indusry_data)
    # print("train_mask:", indusry_data.train_mask)

    # 不加这个可能会报错
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 数据集准备
    # dataset = indusry_data
    # # data = dataset[0]

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
