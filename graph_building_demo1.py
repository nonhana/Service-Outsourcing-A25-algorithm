class Graph:
    def __init__(self, mat, unconn=0):
        # mat是一个二层嵌套的列表，也就是一个矩阵->邻接矩阵，存储记录点和点之间的关系
        vnum = len(mat)  # 顶点个数
        for x in mat:
            if len(x) != vnum:
                raise ValueError("参数错误")
        self._mat = [mat[i][:] for i in range(vnum)]  # 做拷贝
        self._unconn = unconn
        self._vnum = vnum

    # 顶点个数
    def vertex_num(self):
        return self._vnum

    # 顶点是否无效
    def _invalid(self, v):
        return v < 0 or v >= self._vnum

    # 添加边
    def add_edge(self, vi, vj, val=1):
        if self._invalid(vi) or self._invalid(vj):
            raise ValueError(str(vi) + "or" + str(vj) + "不是有效的顶点")

    # 获取边的值
    def get_edge(self, vi, vj):
        if self._invalid(vi) or self._invalid(vj):
            raise ValueError(str(vi) + "or" + str(vj) + "不是有效的顶点")
        return self._mat[vi][vj]

    # 获得一个顶点的各条出边
    def out_edges(self, vi):
        if self._invalid(vi):
            raise ValueError(str(vi) + "不是有效的顶点")
        return self._out_edges(self._mat[vi], self._unconn)

    @staticmethod
    def _out_edges(row, unconn):
        edges = []
        for i in range(len(row)):
            if row[i] != unconn:
                edges.append((i, row[i]))
        return edges

    def __str__(self) -> str:
        return "[\n" + ",\n".join(map(str, self._mat)) + "\n]" + "\nUnconnected:" + str(self._unconn)


class Product_List:
    def __init__(self, info, depth=5):
        self._depth = 5
        for i in info:
            self._totalnodes += len(i)


class Industry:
    def __init__(self, name, cnodes, fnode):
        self._name = name
        self._cnodes = cnodes
        self._fnode = fnode


class Company:
    def __init__(self, name, industry):
        self._name = name
        self._industry = industry


class Production:
    def __init__(self, name, companies, front, behind):
        self._name = name
        self._companies = companies
        self._front = front
        self._behind = behind


class Relationship:
    def __init__(self, name, value, front, behind):
        value = 0
        self._name = name
        self._value = value
        self._front = front
        self._behind = behind


if __name__ == "__main__":
    data = [
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [2, 0, 1, 0]
    ]

    nodes_depth = [[3], [3, 1, 1], [[3, 2, 1], 3, 4]]

    graph1 = Graph(data)
