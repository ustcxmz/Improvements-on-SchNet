from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear


# 无论更新哪个特征，都用LP处理节点特征
# 通过融合节点特征+高斯距离更新边特征
class update_e(torch.nn.Module):
    """
    更新边特征的模块。

    Args:
        hidden_channels (int): 隐藏层的通道数。
        num_filters (int): 滤波器的数量。
        num_gaussians (int): 高斯核的数量。
        cutoff (float): 截止距离，用于计算边的权重。
    """

    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        # 针对节点特征的LP
        self.lin = Linear(hidden_channels, num_filters, bias=False)
        # 针对边（高斯基函数空间中）距离的MLP
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )

        self.reset_parameters()

    def reset_parameters(self):
        """初始化模型参数。"""
        # MLP权重初始化为均匀分布随机数
        # 偏置初始化为0
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):
        """
        前向传播，更新边特征。

        Args:
            v (Tensor): 节点特征。
            dist (Tensor): 边的距离。
            dist_emb (Tensor): 嵌入的距离特征。
            edge_index (Tensor): 边的索引。

        Returns:
            Tensor: 更新后的边特征。
        """
        # 获取目标节点索引
        j, _ = edge_index
        # dist <= cutoff, C(dist)为单调减的截止函数，用来体现距离的影响
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        # 边距离权重
        W = self.mlp(dist_emb) * C.view(-1, 1)
        # 节点特征的LP
        v = self.lin(v)
        # 计入距离的节点特征
        e = v[j] * W
        return e


# 通过融合边特征+MLP更新节点特征
class update_v(torch.nn.Module):
    """
    更新节点特征的模块。

    Args:
        hidden_channels (int): 隐藏层的通道数。
        num_filters (int): 滤波器的数量。
    """

    def __init__(self, hidden_channels, num_filters):
        super(update_v, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1 = Linear(num_filters, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化模型参数。"""
        # MLP权重初始化为均匀分布随机数
        # 偏置初始化为0
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, e, edge_index):
        """
        前向传播，更新节点特征。

        Args:
            v (Tensor): 节点特征。
            e (Tensor): 边特征。
            edge_index (Tensor): 边的索引。

        Returns:
            Tensor: 更新后的节点特征。
        """
        # 用ResNet的方式更新节点特征
        # 获取源节点索引
        _, i = edge_index
        # 将边特征聚合到源节点
        # 修复 scatter_reduce 调用，添加 src 和 reduce 参数
        out = torch.scatter_reduce(src=e, index=i, dim=0, reduce="sum")
        # 两层MLP
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        return v + out


# 融合所有节点特征更新全局特征
class update_u(torch.nn.Module):
    """
    更新全局特征的模块。

    Args:
        hidden_channels (int): 隐藏层的通道数。
        out_channels (int): 输出通道数。
    """

    def __init__(self, hidden_channels, out_channels):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化模型参数。"""
        # MLP权重初始化为均匀分布随机数
        # 偏置初始化为0
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch):
        """
        前向传播，更新全局特征。

        Args:
            v (Tensor): 节点特征。
            batch (Tensor): 批次索引。

        Returns:
            Tensor: 更新后的全局特征。
        """
        # 两层MLP更新节点特征
        v = self.lin1(v)
        v = self.act(v)
        v = self.lin2(v)
        # 计算图的数量
        num_graphs = int(batch.max().item()) + 1
        # 创建形状为 [num_graphs, feature_dim] 的零张量
        u = torch.zeros(num_graphs, v.size(1), device=v.device)
        # 使用 scatter_add_ 实现全局汇聚
        u = u.scatter_add_(0, batch.unsqueeze(-1).expand_as(v), v)
        return u


class emb(torch.nn.Module):
    """
    距离嵌入模块，将距离映射到高斯基函数空间。

    Args:
        start (float): 高斯核的起始值。
        stop (float): 高斯核的结束值。
        num_gaussians (int): 高斯核的数量。
    """

    # 对电子坐标与各高斯核中心计算“高斯距离”，得到电子在高斯基函数下的坐标
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        # 生成均匀分布于指定区间的高斯核中心（一维，后用广播机制进行扩展）
        offset = torch.linspace(start, stop, num_gaussians)
        self.offset = offset
        # 计算高斯函数指数部分的系数
        # 这里的offset[1] - offset[0]是高斯核中心之间的距离
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        # 不对offset进行梯度更新（注册缓冲区）
        self.register_buffer("offset", offset)

    def forward(self, dist):
        """
        前向传播，计算距离的高斯嵌入。

        Args:
            dist (Tensor): 距离。

        Returns:
            Tensor: 嵌入的距离特征。
        """
        # 用广播机制计算距离与高斯核中心之间的距离
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        # 返回高斯函数（坐标）
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    """
    平移的Softplus激活函数。
    """

    # 将值域平移到(-log(2), inf)
    # 平移是为了避免数值不稳定性
    # 其中log(2)是经验数值
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        # 为了得到torch类型的log(2)
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        """
        前向传播，应用激活函数。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: 激活后的张量。
        """
        # softplus(x) = log(1 + exp(x))
        return F.softplus(x) - self.shift


def compute_radius_graph(pos, batch, cutoff):
    """
    手动实现基于半径的邻接关系。

    Args:
        pos (Tensor): 节点位置，形状为 [num_nodes, 3]。
        batch (Tensor): 批次索引，形状为 [num_nodes]。
        cutoff (float): 截止距离。

    Returns:
        Tensor: 边的索引，形状为 [2, num_edges]。
    """
    edge_index = []
    for b in torch.unique(batch):
        mask = batch == b
        pos_b = pos[mask]
        dist = torch.cdist(pos_b, pos_b)  # 计算两两节点之间的欧几里得距离
        src, dst = torch.where((dist <= cutoff) & (dist > 0))  # 排除自身
        edge_index.append(torch.stack([src, dst], dim=0))
    return torch.cat(edge_index, dim=1)


class SchNet(torch.nn.Module):
    """
    SchNet模型的实现，用于量子相互作用建模。

    Args:
        energy_and_force (bool, optional): 是否预测能量和力。默认值为False。
        num_layers (int, optional): 层数。默认值为6。
        hidden_channels (int, optional): 隐藏层通道数。默认值为128。
        out_channels (int, optional): 输出通道数。默认值为1。
        num_filters (int, optional): 滤波器数量。默认值为128。
        num_gaussians (int, optional): 高斯核数量。默认值为50。
        cutoff (float, optional): 截止距离。默认值为10.0。
    """

    def __init__(
        self,
        energy_and_force=False,
        cutoff=10.0,
        num_layers=6,
        hidden_channels=128,
        out_channels=1,
        # 滤波器数量是更新模块中mlp的输出维度，再乘上W得到变化特征向量，最后
        # 将原特征向量与变化特征向量相加(ResNet)得到更新后特征向量
        num_filters=128,  # 滤波器数量越多，mlp的表达能力越强
        num_gaussians=50,  # 高斯核数量越多，嵌入距离特征的精度越高
    ):
        super(SchNet, self).__init__()

        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        # 实例化初始特征向量模块
        self.init_v = Embedding(100, hidden_channels)
        # 实例化高斯嵌入模块
        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        # 实例化节点更新模块(num_layers个)
        self.update_vs = torch.nn.ModuleList(
            [update_v(hidden_channels, num_filters) for _ in range(num_layers)]
        )

        # 实例化边更新模块(num_layers个)
        self.update_es = torch.nn.ModuleList(
            [
                update_e(hidden_channels, num_filters, num_gaussians, cutoff)
                for _ in range(num_layers)
            ]
        )

        # 实例化全局特征更新模块
        self.update_u = update_u(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化模型参数。"""
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, batch_data):
        """
        前向传播，计算模型输出。

        Args:
            z (Tensor): 节点特征。
            pos (Tensor): 节点位置。
            batch (Tensor): 批次索引。

        Returns:
            Tensor: 模型的输出。
        """
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch

        # 使用手动实现的 compute_radius_graph 替代 radius_graph
        edge_index = compute_radius_graph(pos, batch, self.cutoff)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)

        v = self.init_v(z)

        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(v, dist, dist_emb, edge_index)
            v = update_v(v, e, edge_index)

        u = self.update_u(v, batch)
        return u


# backward 函数根据需要，针对全局特征 u 满足的微分方程进行定义
# loss = loss_mlp + loss_pde
