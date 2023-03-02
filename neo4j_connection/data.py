from py2neo import Graph

# 连接neo4j数据库


class MedicalGraph:
    def __init__(self):
        self.g = Graph('http://localhost:7474/', user='neo4j',
                       password='20021209xiang', name='neo4j')

    # 查询

    def query(self):
        # 根节点
        n_node = []
        # 第一条边的相关属性
        r1_startnode = []
        r1_name = []
        r1_endnode = []
        # 第二节点
        m_node = []
        # 第二条边的相关属性
        r2_startnode = []
        r2_name = []
        r2_endnode = []
        # 第三节点
        m1_node = []
        # 第三条边的相关属性
        r3_startnode = []
        r3_name = []
        r3_endnode = []

        # 定义cql语句
        cql = 'match (n:industry {name:"基础化工"})-[r1]-(m)-[r2]-(m1)-[r3]-(m2) return n,r1,m,r2,m1,r3,m2'

        # 查询
        n = self.g.run(cql).data('n')
        r1 = self.g.run(cql).data('r1')
        m = self.g.run(cql).data('m')
        r2 = self.g.run(cql).data('r2')
        m1 = self.g.run(cql).data('m1')
        r3 = self.g.run(cql).data('r3')

        # 根节点
        for i in range(len(n)):
            record = list(n[i].values())
            result = list(record[0].values())[1]
            n_node.append(result)
        # print(n_node)

        # 第一条边
        for i in range(len(r1)):
            record = list(r1[i].values())
            result = str(record[0])
            # 将字符串切片，取出头结点、边名称、尾节点
            r1_startnode.append(result[result.index("(")+1:result.index(")")])
            r1_name.append(result[result.index("[")+2:result.index("]")-3])
            r1_endnode.append(
                result[result.index("(", 5)+1:result.index(")", 10)])
        # print(r1_startnode)
        # print(r1_name)
        # print(r1_endnode)

        # 第二节点
        for i in range(len(m)):
            record = list(m[i].values())
            result = list(record[0].values())[1]
            m_node.append(result)
        # print(m_node)

        # 第二条边
        for i in range(len(r2)):
            record = list(r2[i].values())
            result = str(record[0])
            # 将字符串切片，取出头结点、边名称、尾节点
            r2_startnode.append(result[result.index("(")+1:result.index(")")])
            r2_name.append(result[result.index("[")+2:result.index("]")-3])
            r2_endnode.append(
                result[result.index("(", 5)+1:result.index(")", 10)])
        # print(r2_startnode)
        # print(r2_name)
        # print(r2_endnode)

        # 第三节点
        for i in range(len(m1)):
            record = list(m1[i].values())
            result = list(record[0].values())[1]
            m1_node.append(result)
        # print(m1_node)

        # 第三条边
        for i in range(len(r3)):
            record = list(r3[i].values())
            result = str(record[0])
            # 将字符串切片，取出头结点、边名称、尾节点
            r3_startnode.append(result[result.index("(")+1:result.index(")")])
            r3_name.append(result[result.index("[")+2:result.index("]")-3])
            r3_endnode.append(
                result[result.index("(", 5)+1:result.index(")", 10)])
        # print(r3_startnode)
        # print(r3_name)
        # print(r3_endnode)


if __name__ == '__main__':
    handler = MedicalGraph()
    handler.query()
