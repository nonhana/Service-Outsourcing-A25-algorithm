# 构建网络
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork

# 这个贝叶斯网络中有五个节点: Pollution, Cancer, Smoker, Xray, Dyspnoea.
# ('Pollution', 'Cancer'): 一条有向边, 从 Pollution 指向 Cancer, 表示环境污染有可能导致癌症.
# ('Smoker', 'Cancer'): 吸烟有可能导致癌症.
# ('Cancer', 'Xray'): 得癌症的人可能会去照X射线.
# ('Cancer', 'Dyspnoea'): 得癌症的人可能会呼吸困难.
cancer_model = BayesianNetwork([('Pollution', 'Cancer'),
                                ('Smoker', 'Cancer'),
                                ('Cancer', 'Xray'),
                                ('Cancer', 'Dyspnoea')])


# 设置参数
# 这部分代码主要是建立一些概率表, 然后往表里面填入了一些参数.
# Pollution: 有两种概率, 分别是 0.9 和 0.1.
# Smoker: 有两种概率, 分别是 0.3 和 0.7. (意思是在一个人群里, 有 30% 的人吸烟, 有 70% 的人不吸烟)
# Cancer: envidence 表示有 Smoker 和 Pollution 两个节点指向 Cancer 节点
cpd_pollution = TabularCPD(variable='Pollution', variable_card=2,
                      values=[[0.9], [0.1]])
cpd_smoke = TabularCPD(variable='Smoker', variable_card=2,
                       values=[[0.3], [0.7]])
cpd_cancer = TabularCPD(variable='Cancer', variable_card=2,
                        values=[[0.03, 0.05, 0.001, 0.02],
                                [0.97, 0.95, 0.999, 0.98]],
                        evidence=['Smoker', 'Pollution'],
                        evidence_card=[2, 2])
cpd_xray = TabularCPD(variable='Xray', variable_card=2,
                      values=[[0.9, 0.2], [0.1, 0.8]],
                      evidence=['Cancer'], evidence_card=[2])
cpd_dyspnoea = TabularCPD(variable='Dyspnoea', variable_card=2,
                      values=[[0.65, 0.3], [0.35, 0.7]],
                      evidence=['Cancer'], evidence_card=[2])
cancer_model.add_cpds(cpd_pollution, cpd_smoke, cpd_cancer, cpd_xray, cpd_dyspnoea)


# 测试构建出来的网络结构
print(cancer_model.check_model())


# 变量消除法是精确推断的一种方法.
asia_infer = VariableElimination(cancer_model)
q = asia_infer.query(variables=['Cancer'], evidence={'Smoker': 0})
print(q)
