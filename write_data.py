from py2neo import Graph
import re

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
        # 打开data.txt文件，准备将数据进行写入
        file_handle = open('data涤纶.txt', mode='w')

        # 定义cql语句
        cql = 'match (n:industry {name:"基础化工"})-[r1]-(m1:industry {name:"化学纤维"})-[r2]-(m2:industry)-[r3]-(m3)-[r4]-(m4)-[r5]->(m5)-[r6:`上游材料`]-(m6) return n,r1,m1,r2,m2,r3,m3,r4,m4,r5,m5,r6,m6 limit 300'

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
        self.n_node = list(set(self.n_node))
        for item in self.n_node:
            file_handle.write(item+'\n')
        file_handle.write('===根节点end===\n')
        # print(self.n_node)

        # 第一条边
        file_handle.write('===第一条边start===\n')
        edge_list = []
        for i in range(len(r1)):
            record = list(r1[i].values())
            result = str(record[0])
            edge_list.append(result)
        edge_list = list(set(edge_list))
        # 将字符串切片，取出头结点、边名称、尾节点
        for i in range(len(edge_list)):
            self.r1_startnode.append(
                edge_list[i][edge_list[i].index("(")+1:edge_list[i].index("-")-1])
            string = edge_list[i][edge_list[i].index(
                "[")+2:edge_list[i].index("]")]
            self.r1_name.append(
                re.sub(r"'", "", string)
            )
            self.r1_endnode.append(
                edge_list[i][edge_list[i].index("]")+4:edge_list[i].index(")", len(edge_list[i])-1)])
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
        self.m1_node = list(set(self.m1_node))
        for item in self.m1_node:
            file_handle.write(item+'\n')
        file_handle.write('===第二节点end===\n')
        # print(self.m1_node)

        # 第二条边
        file_handle.write('===第二条边start===\n')
        edge_list = []
        for i in range(len(r2)):
            record = list(r2[i].values())
            result = str(record[0])
            edge_list.append(result)
        edge_list = list(set(edge_list))
        # 将字符串切片，取出头结点、边名称、尾节点
        for i in range(len(edge_list)):
            self.r2_startnode.append(
                edge_list[i][edge_list[i].index("(")+1:edge_list[i].index("-")-1])
            self.r2_name.append(
                edge_list[i][edge_list[i].index("[")+2:edge_list[i].index("]")])
            self.r2_endnode.append(
                edge_list[i][edge_list[i].index("]")+4:edge_list[i].index(")", len(edge_list[i])-1)])
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
        self.m2_node = list(set(self.m2_node))
        for item in self.m2_node:
            file_handle.write(item+'\n')
        file_handle.write('===第二节点end===\n')
        # print(self.m2_node)

        # 第三条边
        file_handle.write('===第三条边start===\n')
        edge_list = []
        for i in range(len(r3)):
            record = list(r3[i].values())
            result = str(record[0])
            edge_list.append(result)
        edge_list = list(set(edge_list))
        # 将字符串切片，取出头结点、边名称、尾节点
        for i in range(len(edge_list)):
            self.r3_startnode.append(
                edge_list[i][edge_list[i].index("(")+1:edge_list[i].index("-")-1])
            self.r3_name.append(
                edge_list[i][edge_list[i].index("[")+2:edge_list[i].index("]")])
            self.r3_endnode.append(
                edge_list[i][edge_list[i].index("]")+4:edge_list[i].index(")", len(edge_list[i])-1)])
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
        self.m3_node = list(set(self.m3_node))
        for item in self.m3_node:
            file_handle.write(item+'\n')
        file_handle.write('===第四节点end===\n')
        # print(self.m3_node)

        # 第四条边
        file_handle.write('===第四条边start===\n')
        edge_list = []
        for i in range(len(r4)):
            record = list(r4[i].values())
            result = str(record[0])
            edge_list.append(result)
        edge_list = list(set(edge_list))
        # 将字符串切片，取出头结点、边名称、尾节点
        for i in range(len(edge_list)):
            self.r4_startnode.append(
                edge_list[i][edge_list[i].index("(")+1:edge_list[i].index("-")-1])
            self.r4_name.append(
                edge_list[i][edge_list[i].index("[")+2:edge_list[i].index("]")])
            self.r4_endnode.append(
                edge_list[i][edge_list[i].index("]")+4:edge_list[i].index(")", len(edge_list[i])-1)])
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
        self.m4_node = list(set(self.m4_node))
        for item in self.m4_node:
            file_handle.write(item+'\n')
        file_handle.write('===第五节点end===\n')
        # print(self.m4_node)

        # 第五条边
        file_handle.write('===第五条边start===\n')
        edge_list = []
        for i in range(len(r5)):
            record = list(r5[i].values())
            result = str(record[0])
            edge_list.append(result)
        edge_list = list(set(edge_list))
        # 将字符串切片，取出头结点、边名称、尾节点
        for i in range(len(edge_list)):
            self.r5_startnode.append(
                edge_list[i][edge_list[i].index("(")+1:edge_list[i].index("-")-1])
            self.r5_name.append(
                edge_list[i][edge_list[i].index("[")+2:edge_list[i].index("]")])
            self.r5_endnode.append(
                edge_list[i][edge_list[i].index("]")+4:edge_list[i].index(")", len(edge_list[i])-1)])
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
        self.m5_node = list(set(self.m5_node))
        for item in self.m5_node:
            file_handle.write(item+'\n')
        file_handle.write('===第六节点end===\n')
        # print(self.m5_node)

        # 第六条边
        file_handle.write('===第六条边start===\n')
        edge_list = []
        for i in range(len(r6)):
            record = list(r6[i].values())
            result = str(record[0])
            edge_list.append(result)
        edge_list = list(set(edge_list))
        # 将字符串切片，取出头结点、边名称、尾节点
        for i in range(len(edge_list)):
            self.r6_startnode.append(
                edge_list[i][edge_list[i].index("(")+1:edge_list[i].index("-")-1])
            self.r6_name.append(
                edge_list[i][edge_list[i].index("[")+2:edge_list[i].index("]")])
            self.r6_endnode.append(
                edge_list[i][edge_list[i].index("]")+4:edge_list[i].index(")", len(edge_list[i])-1)])
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
        self.m6_node = list(set(self.m6_node))
        for item in self.m6_node:
            file_handle.write(item+'\n')
        file_handle.write('===第七节点end===\n')
        # print(self.m6_node)

        # 写入完毕后，关闭文件
        file_handle.close()


if __name__ == "__main__":
    # 连接数据库并读取数据，写到data.txt文件中
    handler = ReadGraph()
    handler.query()
