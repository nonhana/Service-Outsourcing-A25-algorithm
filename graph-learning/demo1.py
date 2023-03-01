from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(f'Dataset:{dataset}:')
print('=' * 30)
print(f'Number of graphs:{len(dataset)}')
print(f'Number of features:{dataset.num_features}')
print(f'Number of classes:{dataset.num_classes}')

print('=' * 30)
data = dataset[0]
# train_mask = [True,False,...] ：代表第1个点是有标签的，第2个点是没标签的，方便后面LOSS的计算
# Data(x=[节点数, 特征数], edge_index=[2, 边的条数], y=[节点数], train_mask=[节点数])
print(data)
