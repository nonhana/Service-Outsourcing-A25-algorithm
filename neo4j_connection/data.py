from py2neo import Graph


class MedicalGraph:
    def __init__(self):
        self.g = Graph('http://localhost:7474/', user='neo4j',
                       password='20021209xiang', name='neo4j')
        


    #查询
    def query(tx, name):
        content = []
        link_list = []
        id_list = []
        # 模糊查询
        statementrecord = tx.run("MATCH (n) WHERE n.name = " + name + " RETURN n")
        # 查询结果存入字典
        for record in statementrecord.records():
            dic = dict(record['n'])
            id_list.append(record['n'].id)
            dic['id'] = record['n'].id
            content.append(dic)
        # 根据id查询关系节点
        linkrecord = tx.run("MATCH (n)<-[r]->(x) WHERE id(n) in {list} RETURN id(n), labels(x), x.name, r", list=id_list)
        dic = {}
        count = 1
        # 存入字典，将id相同的查询结果进行整合
        for record in linkrecord.records():
            idn = record['id(n)']
            xlabels = record['labels(x)'][0].lower()
            xname = record['x.name']
            if dic.get('id') == idn:
                if dic.get(xlabels):
                    if xname not in dic[xlabels]:
                        dic[xlabels].append(xname)
                        count += 1
                else:
                    count += 1
                    dic[xlabels] = [xname]
            else:
                if dic != {}:
                    dic['cnt'] = count
                    count = 1
                    link_list.append(copy.copy(dic))
                dic = {}
                dic['id'] = idn
                dic[xlabels] = [xname]
        dic['cnt'] = count
        link_list.append(dic)
        for link in link_list:
            for con in content:
                try:
                    if con['id'] == link['id']:
                        con.update(link)
                except:
                    pass
        return content

if __name__ == '__main__':
    handler = MedicalGraph()
