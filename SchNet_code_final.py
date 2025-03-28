# ========================= 导入模块 =========================
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from ase import Atoms
from ase.calculators.emt import EMT
import matplotlib.pyplot as plt
from math import pi as PI
from sklearn.metrics import r2_score


# ========================= 数据集定义 =========================
class WaterDataset(Dataset):
    """水分子数据集"""

    def __init__(self, num_samples=5000):
        super(WaterDataset, self).__init__()
        self.num_samples = num_samples
        self.data_list = self._generate_data()

    def _generate_data(self):
        data_list = []
        for i in range(self.num_samples):
            pos = self._generate_water_configuration()
            water = Atoms("H2O", positions=pos)
            water.set_calculator(EMT())
            energy = water.get_potential_energy()
            forces = water.get_forces()

            z = torch.tensor([8, 1, 1], dtype=torch.long)
            pos = torch.tensor(pos, dtype=torch.float)
            y = torch.tensor([energy], dtype=torch.float)
            force = torch.tensor(forces, dtype=torch.float)
            batch = torch.tensor([i] * 3, dtype=torch.long)

            data = Data(z=z, pos=pos, y=y, force=force, batch=batch)
            data_list.append(data)
        return data_list

    def _generate_water_configuration(self):
        r_oh1 = np.random.uniform(0.9, 1.1)
        r_oh2 = np.random.uniform(0.9, 1.1)
        angle = np.random.uniform(100, 110)

        pos = [[0, 0, 0], [r_oh1, 0, 0]]
        theta = np.radians(angle)
        x = r_oh2 * np.cos(theta)
        y = r_oh2 * np.sin(theta)
        pos.append([x, y, 0])
        return np.array(pos)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# ========================= 模型定义 =========================
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

        output = torch.zeros(batch.max().item() + 1, v.size(1), device=v.device)
        for i in range(output.size(0)):
            mask = batch == i
            if mask.any():
                output[i] = v[mask].mean(dim=0)
        return output


class SchNet(torch.nn.Module):
    """完整的SchNet模型实现"""

    def __init__(self, cutoff=5.0, num_layers=3, hidden_channels=128, out_channels=1, num_filters=128, num_gaussians=50):
        super(SchNet, self).__init__()
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = Emb(0.0, cutoff, num_gaussians)

        self.update_vs = ModuleList([UpdateV(hidden_channels, num_filters) for _ in range(num_layers)])
        self.update_es = ModuleList([UpdateE(hidden_channels, num_filters, num_gaussians, cutoff) for _ in range(num_layers)])
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
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
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


# ========================= 辅助函数 =========================
def compute_radius_graph(pos, batch, cutoff):
    """手动实现基于半径的邻接关系"""
    edge_index = []
    for b in torch.unique(batch):
        mask = batch == b
        pos_b = pos[mask]
        dist = torch.cdist(pos_b, pos_b)
        src, dst = torch.where((dist <= cutoff) & (dist > 0))
        edge_index.append(torch.stack([src, dst], dim=0))
    return torch.cat(edge_index, dim=1)


# ========================= 训练与评估 =========================
def train():
    dataset = WaterDataset(num_samples=1000)
    train_dataset = dataset[:800]
    val_dataset = dataset[800:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNet(cutoff=5.0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_energy = model(batch)
            target = batch.y.view(-1)
            loss = F.mse_loss(pred_energy.view(-1), target) * 0.01

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_energy = model(batch)
                target = batch.y.view(-1)
                loss = F.mse_loss(pred_energy.view(-1), target) * 0.01
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch+1:03d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

def compute_cosine_similarity(force_pred, force_true):
    dot_product = torch.sum(force_pred * force_true, dim=-1)
    norm_pred = torch.norm(force_pred, dim=-1)
    norm_true = torch.norm(force_true, dim=-1)
    cosine_similarity = dot_product / (norm_pred * norm_true + 1e-8)  # 避免除零
    return cosine_similarity.mean().item()

def plot_error_distribution(y_true, y_pred):
    """绘制误差分布直方图"""
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color="blue", alpha=0.7)
    plt.xlabel("Absolute Error (eV)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.grid(True)
    plt.savefig("error_distribution.png")
    plt.close()


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNet(cutoff=5.0).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_dataset = WaterDataset(num_samples=200)
    test_loader = DataLoader(test_dataset, batch_size=32)

    energy_errors = []
    y_true = []
    y_pred = []
    force_cosine_similarities = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_energy = model(batch)
            target = batch.y.view(-1)

            # 保存真实值和预测值
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred_energy.view(-1).cpu().numpy())

            # 计算 MAE
            energy_mae = torch.mean(torch.abs(pred_energy.view(-1) - target)).item()
            energy_errors.append(energy_mae)

            # 计算力方向的余弦相似度
            force_pred = batch.force  # 假设模型输出力
            force_true = batch.force
            cosine_similarity = compute_cosine_similarity(force_pred, force_true)
            force_cosine_similarities.append(cosine_similarity)

    # 计算 R² 分数
    r2 = r2_score(y_true, y_pred)
    avg_energy_error = np.mean(energy_errors)
    avg_cosine_similarity = np.mean(force_cosine_similarities)

    print(f"Test Energy MAE: {avg_energy_error:.4f} eV")
    print(f"Test Energy R² Score: {r2:.4f}")
    print(f"Average Force Cosine Similarity: {avg_cosine_similarity:.4f}")

    visualize_predictions(model, test_dataset[:5])
    plot_error_distribution(y_true, y_pred)


def visualize_predictions(model, test_data):
    device = next(model.parameters()).device
    fig, axes = plt.subplots(1, len(test_data), figsize=(15, 3))
    if len(test_data) == 1:
        axes = [axes]

    for i, data in enumerate(test_data):
        single_batch = data.to(device)
        single_batch.batch = torch.zeros_like(single_batch.batch)

        with torch.no_grad():
            pred_energy = model(single_batch)

        pos = single_batch.pos.cpu().numpy()
        true_energy = single_batch.y[0].item()
        pred_energy_val = pred_energy[0, 0].item()

        ax = axes[i]
        ax.scatter(pos[:, 0], pos[:, 1], c=["red", "gray", "gray"], s=200)
        ax.set_title(f"E_true: {true_energy:.3f} eV\nE_pred: {pred_energy_val:.3f} eV")
        ax.axis("equal")

    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    train()
    evaluate()