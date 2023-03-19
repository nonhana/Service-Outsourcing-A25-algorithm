import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class DataSource:
    def __init__(self):
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

        file_handler = open('data改性塑料.txt', mode='r')
        node_num = 0
        node_flag = False
        edge_num = 0
        edge_flag = 0
        for line in file_handler:
            # 设置读取状态
            if line.strip().find('节点start===') != -1:
                node_num = node_num+1
                node_flag = True
            if line.strip().find('节点end===') != -1:
                node_flag = False
            if line.strip().find('边start===') != -1:
                edge_num = edge_num+1
                edge_flag = 1
            if edge_flag > 0 and line.strip() == '':
                edge_flag = edge_flag+1
            if line.strip().find('边end===') != -1:
                edge_flag = 0

            # 读取节点
            if node_flag and line.strip() != '' and line.strip().find('节点end===') == -1 and line.strip().find('节点start===') == -1:
                if node_num == 1:
                    if line.strip() not in self.n_node:
                        self.n_node.append(line.strip())
                if node_num == 2:
                    if line.strip() not in self.m1_node:
                        self.m1_node.append(line.strip())
                if node_num == 3:
                    if line.strip() not in self.m2_node:
                        self.m2_node.append(line.strip())
                if node_num == 4:
                    if line.strip() not in self.m3_node:
                        self.m3_node.append(line.strip())
                if node_num == 5:
                    if line.strip() not in self.m4_node:
                        self.m4_node.append(line.strip())
                if node_num == 6:
                    if line.strip() not in self.m5_node:
                        self.m5_node.append(line.strip())
                if node_num == 7:
                    if line.strip() not in self.m6_node:
                        self.m6_node.append(line.strip())
            # 读取边
            if edge_flag == 1 and line.strip() != '' and line.strip().find('边end===') == -1 and line.strip().find('边start===') == -1:
                if edge_num == 1:
                    self.r1_startnode.append(line.strip())
                if edge_num == 2:
                    self.r2_startnode.append(line.strip())
                if edge_num == 3:
                    self.r3_startnode.append(line.strip())
                if edge_num == 4:
                    self.r4_startnode.append(line.strip())
                if edge_num == 5:
                    self.r5_startnode.append(line.strip())
                if edge_num == 6:
                    self.r6_startnode.append(line.strip())
            if edge_flag == 2 and line.strip() != '' and line.strip().find('边end===') == -1 and line.strip().find('边start===') == -1:
                if edge_num == 1:
                    self.r1_name.append(line.strip())
                if edge_num == 2:
                    self.r2_name.append(line.strip())
                if edge_num == 3:
                    self.r3_name.append(line.strip())
                if edge_num == 4:
                    self.r4_name.append(line.strip())
                if edge_num == 5:
                    self.r5_name.append(line.strip())
                if edge_num == 6:
                    self.r6_name.append(line.strip())
            if edge_flag == 3 and line.strip() != '' and line.strip().find('边end===') == -1 and line.strip().find('边start===') == -1:
                if edge_num == 1:
                    self.r1_endnode.append(line.strip())
                if edge_num == 2:
                    self.r2_endnode.append(line.strip())
                if edge_num == 3:
                    self.r3_endnode.append(line.strip())
                if edge_num == 4:
                    self.r4_endnode.append(line.strip())
                if edge_num == 5:
                    self.r5_endnode.append(line.strip())
                if edge_num == 6:
                    self.r6_endnode.append(line.strip())


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
        # 邻接矩阵
        self.matrix = []
        # 总的节点个数
        self.numVertices = 0
        self.visble = nx.Graph()
        # 特征向量
        self.feature_vector = []
        # 记录第n个节点的位置n
        self.labels = []
        # 节点名称数组
        self.name_labels = []
        # 边矩阵
        self.edge_matrix = []

    # 增加顶点，具有属性：id，类型，名称
    def addVertex(self, key, type, name):
        self.labels.append(self.numVertices)
        if name not in self.name_labels:
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

    # 计算节点的中心度作为特征值

    def eigenvectorCentrality(self):
        eigenvector = nx.eigenvector_centrality(self.visble)
        for item in eigenvector:
            list = []
            list.append(eigenvector[item])
            self.feature_vector.append(list)
        return


if __name__ == "__main__":
    # 连接数据库并读取数据
    handler = DataSource()

    # 添加顶点
    g = IndustryGraph()
    # 根节点
    g.addVertex(g.getVertices(), "industry", handler.n_node[0])
    # 一级产业
    for item in handler.m1_node:
        if item not in g.name_labels:
            g.addVertex(g.getVertices(), "industry", item)
    # 二级产业
    for item in handler.m2_node:
        if item not in g.name_labels:
            g.addVertex(g.getVertices(), "industry", item)
    # 公司
    for item in handler.m3_node:
        if item not in g.name_labels:
            g.addVertex(g.getVertices(), "company", item)
    # 主营产品
    for item in handler.m4_node:
        if item not in g.name_labels:
            g.addVertex(g.getVertices(), "product", item)
    # 产品小类
    for item in handler.m5_node:
        if item not in g.name_labels:
            g.addVertex(g.getVertices(), "littleproduct", item)
    # 上游材料
    for item in handler.m6_node:
        if item not in g.name_labels:
            g.addVertex(g.getVertices(), "material", item)

    # 初始化邻接矩阵
    g.initMatrix(g.getVertices())

    # 添加边
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
    print(g.matrix)

    # 画出拓扑结构
    # nx.draw(g.visble, node_size=100, node_color="skyblue", with_labels=True)
    # plt.show()

    # 计算节点特征值
    g.eigenvectorCentrality()
    print(g.feature_vector)

    print(len(g.name_labels))
