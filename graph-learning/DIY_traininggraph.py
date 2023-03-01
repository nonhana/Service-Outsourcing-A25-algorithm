import torch
from torch_geometric.data import Data

if __name__ == '__main__':
    # 定义节点特征向量x和标签y
    x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

    # 定义边
    edge_index = torch.tensor([[0, 1, 2, 0, 3],  # 起始点
                               [1, 0, 1, 3, 2]], dtype=torch.long)  # 终止点

    # 定义train_mask
    train_mask = [(True if d is not None else False) for d in y]

    # 构建data
    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask)
    print("data:", data)
    print("train_mask:", data.train_mask)
