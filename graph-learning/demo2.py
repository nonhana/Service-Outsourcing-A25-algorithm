import os
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt


# 画图函数
def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


# 画点函数
def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch:{epoch},Loss:{loss.item():.4f}', fontsize=16)
    plt.show()


if __name__ == '__main__':
    # 不加这个可能会报错
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)
