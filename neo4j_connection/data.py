from py2neo import Graph
import numpy as np
import networkx as nx


# 连接neo4j数据库


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
        self.m_node = []
        # 第二条边的相关属性
        self.r2_startnode = []
        self.r2_name = []
        self.r2_endnode = []
        # 第三节点
        self.m1_node = []
        # 第三条边的相关属性
        self.r3_startnode = []
        self.r3_name = []
        self.r3_endnode = []
        # 第四节点
        self.m2_node = []
        # 第四条边的相关属性
        self.r4_startnode = []
        self.r4_name = []
        self.r4_endnode = []
        # 第五节点
        self.m3_node = []

    # 查询

    def query(self):

        # 定义cql语句
        cql = 'match (n:industry {name:"基础化工"})-[r1]-(m:industry {name:"塑料"})-[r2]-(m1:industry {name:"膜材料"})-[r3]-(m2)-[r4]-(m3) return n,r1,m,r2,m1,r3,m2,r4,m3'

        # 查询
        n = self.g.run(cql).data('n')
        r1 = self.g.run(cql).data('r1')
        m = self.g.run(cql).data('m')
        r2 = self.g.run(cql).data('r2')
        m1 = self.g.run(cql).data('m1')
        r3 = self.g.run(cql).data('r3')
        m2 = self.g.run(cql).data('m2')
        r4 = self.g.run(cql).data('r4')
        m3 = self.g.run(cql).data('m3')

        # 根节点
        for i in range(len(n)):
            record = list(n[i].values())
            result = list(record[0].values())[1]
            self.n_node.append(result)
        self.n_node = list(set(self.n_node))
        print(self.n_node)

        # 第一条边
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
        # print(self.r1_startnode)
        # print(self.r1_name)
        # print(self.r1_endnode)

        # 第二节点
        for i in range(len(m)):
            record = list(m[i].values())
            result = list(record[0].values())[1]
            self.m_node.append(result)
        self.m_node = list(set(self.m_node))
        # print(self.m_node)

        # 第二条边
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
        # print(self.r2_startnode)
        # print(self.r2_name)
        # print(self.r2_endnode)

        # 第三节点
        for i in range(len(m1)):
            record = list(m1[i].values())
            result = list(record[0].values())[1]
            self.m1_node.append(result)
        self.m1_node = list(set(self.m1_node))
        # print(self.m1_node)

        # 第三条边
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
        # print(self.r3_startnode)
        # print(self.r3_name)
        # print(self.r3_endnode)

        # 第四节点
        for i in range(len(m2)):
            record = list(m2[i].values())
            result = list(record[0].values())[1]
            self.m2_node.append(result)
        self.m2_node = list(set(self.m2_node))
        # print(self.m2_node)

        # 第四条边
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
        # print(self.r4_startnode)
        # print(self.r4_name)
        # print(self.r4_endnode)

        # 第五节点
        for i in range(len(m3)):
            record = list(m3[i].values())
            result = list(record[0].values())[0]
            self.m3_node.append(result)
        self.m3_node = list(set(self.m3_node))
        # print(self.m3_node)


if __name__ == '__main__':
    handler = ReadGraph()
    handler.query()
