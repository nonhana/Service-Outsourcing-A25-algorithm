# gcnmodel_create

本文件夹为构建、训练、测试、导出GCN模型的代码，其中：

- gcn_model.pth为提前训练完成而导出的二进制GCN模型，用于test_model.py中的测试。
- write_data为从Neo4j数据库中以一定的Cypher语句读出符合定义的产业链数据，并将其保存为txt文件，以便后续的使用。
- industry_modelbuilding为构建、训练、导出GCN模型的核心代码，主要利用从write_data中读出的txt数据文件进行模型的训练，训练完毕后将模型进行导出，存储为二进制的gcn_model.pth文件。
- txt_datas中保存了多个从write_data读出的txt数据文件，用户可以自行采用进行模型的训练，或是自己重新构建新的训练数据集。

