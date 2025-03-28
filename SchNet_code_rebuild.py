import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from ase import Atoms
from ase.calculators.emt import EMT


class WaterDataset(Dataset):
    """水分子数据集"""

    def __init__(self, num_samples=5000):
        super(WaterDataset, self).__init__()
        self.num_samples = num_samples
        self.data_list = self._generate_data()

    def _generate_data(self):
        data_list = []
        for i in range(self.num_samples):
            # 随机生成水分子构型
            pos = self._generate_water_configuration()

            # 使用ASE的EMT计算器计算能量和力 (简化计算，实际应用应使用DFT)
            water = Atoms("H2O", positions=pos)
            water.set_calculator(EMT())
            energy = water.get_potential_energy()
            forces = water.get_forces()

            # 转换为PyTorch Geometric的Data对象
            z = torch.tensor([8, 1, 1], dtype=torch.long)  # 原子序数 [O, H, H]
            pos = torch.tensor(pos, dtype=torch.float)
            y = torch.tensor([energy], dtype=torch.float)
            force = torch.tensor(forces, dtype=torch.float)
            batch = torch.tensor([i] * 3, dtype=torch.long)  # 添加批次索引

            data = Data(z=z, pos=pos, y=y, force=force, batch=batch)
            data_list.append(data)

        return data_list

    def _generate_water_configuration(self):
        """生成合理的水分子构型"""
        # O-H键长在0.9-1.1 Å之间
        r_oh1 = np.random.uniform(0.9, 1.1)
        r_oh2 = np.random.uniform(0.9, 1.1)
        # H-O-H键角在100-110度之间
        angle = np.random.uniform(100, 110)

        # 第一个H原子在x轴上
        pos = [[0, 0, 0], [r_oh1, 0, 0]]  # O原子在原点  # H1

        # 第二个H原子在xy平面
        theta = np.radians(angle)
        x = r_oh2 * np.cos(theta)
        y = r_oh2 * np.sin(theta)
        pos.append([x, y, 0])  # H2

        return np.array(pos)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList


class ShiftedSoftplus(torch.nn.Module):
    """平移的Softplus激活函数"""

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Emb(torch.nn.Module):
    """距离嵌入模块"""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(Emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class UpdateE(torch.nn.Module):
    """更新边特征的模块"""

    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(UpdateE, self).__init__()
        self.cutoff = cutoff
        self.lin = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(dist_emb) * C.view(-1, 1)
        v = self.lin(v)
        e = v[j] * W
        return e


class UpdateV(torch.nn.Module):
    """更新节点特征的模块"""

    def __init__(self, hidden_channels, num_filters):
        super(UpdateV, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1 = Linear(num_filters, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, e, edge_index):
        _, i = edge_index
        # 修复 scatter_add_ 调用
        out = torch.zeros_like(v).scatter_add_(dim=0, index=i.unsqueeze(-1).expand_as(e), src=e)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        return v + out


class UpdateU(torch.nn.Module):
    """更新全局特征的模块"""

    def __init__(self, hidden_channels, out_channels):
        super(UpdateU, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch):
        v = self.lin1(v)
        v = self.act(v)
        v = self.lin2(v)

        # 修复：确保 batch 索引范围与 v 的第一个维度大小一致
        max_index = v.size(0)
        batch = batch.clamp(max=max_index - 1)

        u = torch.zeros_like(v).scatter_add_(dim=0, index=batch.unsqueeze(-1).expand_as(v), src=v)
        return u


class SchNet(torch.nn.Module):
    """完整的SchNet模型实现"""

    def __init__(
        self,
        energy_and_force=False,  # 移除与力相关的功能
        cutoff=5.0,
        num_layers=3,
        hidden_channels=128,
        out_channels=1,
        num_filters=128,
        num_gaussians=50,
    ):
        super(SchNet, self).__init__()
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # 初始化模块
        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = Emb(0.0, cutoff, num_gaussians)

        # 交互模块
        self.update_vs = ModuleList(
            [UpdateV(hidden_channels, num_filters) for _ in range(num_layers)]
        )
        self.update_es = ModuleList(
            [
                UpdateE(hidden_channels, num_filters, num_gaussians, cutoff)
                for _ in range(num_layers)
            ]
        )

        # 全局特征更新
        self.update_u = UpdateU(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
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
            batch_data (Data): 包含节点特征、位置和批次索引的批次数据。

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


from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt


def train():
    # 1. 准备数据
    dataset = WaterDataset(num_samples=1000)
    train_dataset = dataset[:800]
    val_dataset = dataset[800:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 2. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNet(cutoff=5.0).to(device)

    # 3. 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    def compute_loss(pred_energy, true_energy, rho=0.01):
        """
        计算损失函数。

        Args:
            pred_energy (Tensor): 预测的能量。
            true_energy (Tensor): 真实的能量。
            rho (float): 能量损失的权重。

        Returns:
            Tuple[Tensor, Tensor]: 总损失、能量损失。
        """
        # 能量损失
        energy_loss = F.mse_loss(pred_energy, true_energy)

        # 总损失
        total_loss = rho * energy_loss
        return total_loss, energy_loss

    # 4. 训练循环
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs = 10

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_energy = model(batch)
            loss, e_loss = compute_loss(
                pred_energy, batch.y
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_energy = model(batch)
                loss, _ = compute_loss(pred_energy, batch.y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch+1:03d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

def evaluate():
    """
    评估模型性能。
    """
    # 加载最佳模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNet(cutoff=5.0).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # 创建测试数据集
    test_dataset = WaterDataset(num_samples=200)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 评估指标
    energy_errors = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_energy = model(batch)

            # 计算MAE
            energy_mae = torch.mean(torch.abs(pred_energy - batch.y)).item()
            energy_errors.append(energy_mae)

    avg_energy_error = np.mean(energy_errors)

    print(f"Test Energy MAE: {avg_energy_error:.4f} eV")

    # 可视化一些预测结果
    visualize_predictions(model, test_dataset[:5])

def visualize_predictions(model, test_data):
    """可视化预测结果"""
    device = next(model.parameters()).device

    fig, axes = plt.subplots(1, len(test_data), figsize=(15, 3))
    if len(test_data) == 1:
        axes = [axes]

    for i, data in enumerate(test_data):
        data = data.to(device)
        with torch.no_grad():
            pred_energy = model(data)

            # 修复：确保 pred_energy 是标量
            pred_energy = pred_energy.mean().item()

        # 转换为numpy
        pos = data.pos.cpu().numpy()

        # 绘制分子构型
        ax = axes[i]
        ax.scatter(pos[:, 0], pos[:, 1], c=["red", "gray", "gray"], s=200)

        ax.set_title(
            f"E_true: {data.y.item():.3f} eV\nE_pred: {pred_energy:.3f} eV"
        )
        ax.axis("equal")

    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.close()


if __name__ == "__main__":
    train()
    evaluate()
