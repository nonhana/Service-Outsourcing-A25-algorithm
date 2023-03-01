from py2neo import Graph  # 这是操作neo4j的库
# 连接数据库
'''
这里是上传数据，有小伙伴不会的可以看我上边给出的CQL语句，把CQL搞到g.run()里边就可以了
'''
g = Graph('http://localhost:7474/', user='neo4j',
          password='20021209xiang', name='neo4j')


def demo():
    g.run(
        "CREATE (:company{name:'北新建材', fullname:'北新集团建材股份有限公司', code:'000786.SZ', location:'深圳证券交易所', time:'1997-06-06'})")


demo()
