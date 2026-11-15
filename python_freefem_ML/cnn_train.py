import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import griddata
import pyvista as pv
import csv


# ==================== CNN 模型定义 ====================

class ConvBlock(nn.Module):
    """基础卷积块：Conv2d + BatchNorm + ReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """下采样块：MaxPool + 卷积"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    """上采样块：ConvTranspose2d + skip concat"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBlock(out_ch + skip_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FlowCNN2d(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, base_width=32):
        super().__init__()
        self.inc = ConvBlock(in_channels, base_width)                  # 64×64, 32ch
        self.down1 = DownBlock(base_width, base_width * 2)             # 32×32, 64ch
        self.down2 = DownBlock(base_width * 2, base_width * 4)         # 16×16, 128ch
        self.down3 = DownBlock(base_width * 4, base_width * 8)         # 8×8,   256ch

        self.bottleneck = nn.Sequential(
            ConvBlock(base_width * 8, base_width * 16),
            ConvBlock(base_width * 16, base_width * 16),
        )

        # ★ 修复：skip_ch 用对应 down 层的实际输出通道数
        self.up3 = UpBlock(base_width * 16, base_width * 4, base_width * 8)   # skip=x3=128
        self.up2 = UpBlock(base_width * 8,  base_width * 2, base_width * 4)   # skip=x2=64
        self.up1 = UpBlock(base_width * 4,  base_width,     base_width * 2)   # skip=x1=32

        self.outc = nn.Conv2d(base_width * 2, out_channels, kernel_size=1)

        # ★ p 通道线性输出（不压 Tanh），ux/uy 用 Tanh
        # 简单起见：全部 Tanh，但 p 权重在 loss 里调低（见训练部分）
        self.activation = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        bottleneck = self.bottleneck(x4)

        x = self.up3(bottleneck, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        x = self.outc(x)
        x = self.activation(x)
        return x


# ==================== train_CNN 类 ====================

class train_CNN:
    def __init__(self, vtk_dir, resolution=64, base_width=32, epochs=200):
        self.vtk_dir = vtk_dir
        self.resolution = resolution
        self.base_width = base_width
        self.epochs = epochs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.X = None
        self.Y = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.model = None
        self.train_losses = []
        self.val_losses = []

    # ---- 插值辅助 ----
    def _interp_field(self, points, vals, grid_points):
        g_nearest = griddata(points, vals, grid_points, method="nearest").reshape(
            self.resolution, self.resolution
        )
        g_cubic = griddata(points, vals, grid_points, method="cubic").reshape(
            self.resolution, self.resolution
        )
        g = np.where(np.isnan(g_cubic), g_nearest, g_cubic)
        return np.nan_to_num(g, nan=0.0)

    # ---- 读单个 VTK ----
    def load_single_case(self, vtk_path, params_override=None):
        mesh = pv.read(vtk_path)
        mesh = mesh.cell_data_to_point_data()

        point_data = mesh.point_data

        if "Velocity" in point_data:
            vel = point_data["Velocity"]
        elif "U" in point_data:
            vel = point_data["U"]
        else:
            raise KeyError(f"找不到速度场，可用字段: {list(point_data.keys())}")

        if "Pressure" in point_data:
            p = point_data["Pressure"]
        elif "p" in point_data:
            p = point_data["p"]
        else:
            raise KeyError(f"找不到压力场，可用字段: {list(point_data.keys())}")

        ux = vel[:, 0]
        uy = vel[:, 1]
        points = mesh.points[:, :2]

        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        xi = np.linspace(x_min, x_max, self.resolution)
        yi = np.linspace(y_min, y_max, self.resolution)
        x_grid, y_grid = np.meshgrid(xi, yi, indexing="ij")
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

        ux_g = self._interp_field(points, ux, grid_points)
        uy_g = self._interp_field(points, uy, grid_points)
        p_g = self._interp_field(points, p, grid_points)

        x_grid = (x_grid - x_min) / (x_max - x_min) * 2 - 1
        y_grid = (y_grid - y_min) / (y_max - y_min) * 2 - 1

        # 参数
        param_file = vtk_path[:-4] + ".csv"
        param_file = param_file.replace("case", "param")
        with open(param_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nu_val = float(row["nu"])
                rho_val = float(row["rho"])
                inlet_val = float(row["inlet"])

        X_single = np.stack([
            x_grid, y_grid,
            np.full_like(x_grid, nu_val),
            np.full_like(x_grid, rho_val),
            np.full_like(x_grid, inlet_val),
        ], axis=-1).astype(np.float32)

        Y_single = np.stack([ux_g, uy_g, p_g], axis=-1).astype(np.float32)

        return X_single, Y_single

    # ---- 批量构建 ----
    def build_dataset(self):
        files = sorted(glob.glob(os.path.join(self.vtk_dir, "case_*.vtk")))
        if len(files) == 0:
            raise FileNotFoundError(f"No case_*.vtk found in {self.vtk_dir}")
        print(f"Found {len(files)} VTK files.")

        X_list, Y_list = [], []
        for i, f in enumerate(files):
            try:
                X_s, Y_s = self.load_single_case(f)
                X_list.append(X_s)
                Y_list.append(Y_s)
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(files)}")
            except Exception as e:
                print(f"  Skipped {f}: {e}")

        self.X = np.array(X_list).astype(np.float32)
        self.Y = np.array(Y_list).astype(np.float32)
        print(f"Dataset built: X={self.X.shape}, Y={self.Y.shape}")
        return self.X, self.Y

    # ---- 构建模型 ----
    def build_model(self):
        self.model = FlowCNN2d(
            in_channels=5, out_channels=3, base_width=self.base_width
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"CNN model built: {n_params:,} trainable parameters")
        return self.model

    # ---- 训练 ----
    def train(self, batch_size=16, lr=1e-3, val_split=0.2):
        if self.X is None or self.Y is None:
            raise ValueError("Call build_dataset() first!")
        if self.model is None:
            self.build_model()

        # ============================================================
        # ★ 调参旋钮区
        # ============================================================
        GRAD_CLIP = 1.0
        WEIGHT_P = 0.1       # ★ p 场 std 74.87，降权避免主导梯度
        WEIGHT_UX = 1.0
        WEIGHT_UY = 0.0      # ★ uy std ≈ 0，不训
        LR = lr
        EARLY_STOP_PATIENCE = 80
        # ============================================================

        # ---- 归一化 ----
        self.x_mean = np.mean(self.X, axis=(0, 1, 2), keepdims=True)
        self.x_std = np.std(self.X, axis=(0, 1, 2), keepdims=True) + 1e-8
        X_norm = (self.X - self.x_mean) / self.x_std

        self.y_mean = np.mean(self.Y, axis=(0, 1, 2), keepdims=True)
        self.y_std = np.std(self.Y, axis=(0, 1, 2), keepdims=True) + 1e-8

        # ★ uy std 保护：太小就设为 1.0，避免 0/0
        self.y_std = np.where(self.y_std < 1e-6, 1.0, self.y_std)

        Y_norm = (self.Y - self.y_mean) / self.y_std

        print(f"X norm range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")
        print(f"Y norm range: [{Y_norm.min():.2f}, {Y_norm.max():.2f}]")
        print(f"Y std per channel (ux, uy, p): {self.y_std.flatten()}")

        # ---- train/val split ----
        N = X_norm.shape[0]
        indices = np.random.permutation(N)
        n_val = int(N * val_split)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        # ★ CNN 需要 (N, C, H, W)
        X_norm_cnn = X_norm.transpose(0, 3, 1, 2)
        Y_norm_cnn = Y_norm.transpose(0, 3, 1, 2)

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_norm_cnn[train_idx]).float(),
                torch.from_numpy(Y_norm_cnn[train_idx]).float()
            ),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_norm_cnn[val_idx]).float(),
                torch.from_numpy(Y_norm_cnn[val_idx]).float()
            ),
            batch_size=batch_size, shuffle=False
        )

        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )

        self.model.train()
        self.train_losses = []
        self.val_losses = []

        best_val = float('inf')
        patience_counter = 0

        print(f"Starting training for {self.epochs} epochs...")
        print(f"  grad_clip={GRAD_CLIP}, lr={LR}, "
              f"w_ux={WEIGHT_UX}, w_uy={WEIGHT_UY}, w_p={WEIGHT_P}")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_loss_ux = 0.0
            epoch_loss_uy = 0.0
            epoch_loss_p = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(xb)

                loss_all = criterion(y_pred, yb).mean(dim=(2, 3))
                loss_ux = loss_all[:, 0].mean()
                loss_uy = loss_all[:, 1].mean()
                loss_p = loss_all[:, 2].mean()

                loss = WEIGHT_UX * loss_ux + WEIGHT_UY * loss_uy + WEIGHT_P * loss_p

                loss.backward()

                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=GRAD_CLIP
                    )

                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_ux += (WEIGHT_UX * loss_ux).item()
                epoch_loss_uy += (WEIGHT_UY * loss_uy).item()
                epoch_loss_p += (WEIGHT_P * loss_p).item()

            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            avg_ux = epoch_loss_ux / len(train_loader)
            avg_uy = epoch_loss_uy / len(train_loader)
            avg_p = epoch_loss_p / len(train_loader)
            self.train_losses.append(avg_loss)

            # val
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    y_pred = self.model(xb)
                    val_loss += criterion(y_pred, yb).mean().item()
            self.model.train()
            avg_val = val_loss / len(val_loader)
            self.val_losses.append(avg_val)

            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"  Epoch [{epoch + 1}/{self.epochs}] "
                f"Train: {avg_loss:.4f} "
                f"(ux={avg_ux:.4f}, uy={avg_uy:.4f}, p={avg_p:.4f}) "
                f"Val: {avg_val:.4f} "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
                f"{' ← best' if avg_val == best_val else ''}"
            )

            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}, best val={best_val:.4f}")
                break

        print("Training finished.")
        return self.train_losses

    # ---- 保存 / 加载 ----
    def save_model(self, path="cnn_ns_model.pth"):
        if self.model is None:
            raise ValueError("No model to save.")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
        }, path)
        print(f"Model + normalization stats saved to {path}")

    def load_model(self, path="cnn_ns_model.pth"):
        if self.model is None:
            self.build_model()
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.x_mean = ckpt.get('x_mean')
        self.x_std = ckpt.get('x_std')
        self.y_mean = ckpt.get('y_mean')
        self.y_std = ckpt.get('y_std')
        print(f"Model loaded from {path}")

    # ---- 推理 ----
    def predict(self, X_input):
        if self.model is None:
            raise ValueError("Call build_model() first.")
        if self.x_mean is None or self.y_mean is None:
            raise ValueError("No normalization stats. Train or load_model() first.")

        self.model.eval()
        with torch.no_grad():
            if X_input.ndim == 3:
                X_input = X_input[np.newaxis, ...]

            X_input_t = X_input.transpose(0, 3, 1, 2)

            X_norm = (X_input_t - self.x_mean) / self.x_std
            X_tensor = torch.from_numpy(X_norm).float().to(self.device)
            y_pred_norm = self.model(X_tensor).cpu().numpy()
            y_pred_norm = y_pred_norm.transpose(0, 2, 3, 1)
            y_pred = y_pred_norm * self.y_std + self.y_mean

        return y_pred.squeeze(0) if X_input.shape[0] == 1 else y_pred


# ==================== 主程序 ====================

if __name__ == "__main__":

    vtk_dir = r"D:\FreeFEM_file\run_100file\result"

    # ---- 诊断 VTK 结构 ----
    print("=" * 60)
    print("Diagnosing VTK structure (first 5 cases)...")
    print("=" * 60)
    files = sorted(glob.glob(os.path.join(vtk_dir, "case_*.vtk")))
    for f in files[:5]:
        mesh = pv.read(f)
        print(f"\n{os.path.basename(f)}:")
        print(f"  n_points={mesh.n_points}, n_cells={mesh.n_cells}")
        print(f"  point_data keys: {list(mesh.point_data.keys())}")
        print(f"  cell_data keys: {list(mesh.cell_data.keys())}")

    # ---- 训练 CNN ----
    print("\n" + "=" * 60)
    print("Training CNN (U-Net style)...")
    print("=" * 60)

    cnn = train_CNN(vtk_dir=vtk_dir, resolution=64, base_width=32, epochs=200)
    cnn.build_dataset()
    cnn.build_model()
    losses = cnn.train(batch_size=16, lr=1e-3, val_split=0.2)

    # 保存
    cnn.save_model("cnn_ns_model.pth")

    # 测试预测
    test_input = cnn.X[0]
    pred = cnn.predict(test_input)
    print(f"\nPrediction shape: {pred.shape}")
    print(f"Pred ux max: {pred[:, :, 0].max():.4f}")
    print(f"True  ux max: {cnn.Y[0, :, :, 0].max():.4f}")
    print(f"Pred p max: {pred[:, :, 2].max():.4f}")
    print(f"True  p max: {cnn.Y[0, :, :, 2].max():.4f}")

    print("\nALL DONE.")