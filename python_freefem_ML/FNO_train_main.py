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


# ==================== FNO 模型定义 ====================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * 0.5)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels,
            x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    def __init__(self, modes1=6, modes2=6, width=16, in_channels=5, out_channels=3):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(in_channels, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = nn.functional.relu(self.conv0(x) + self.w0(x))
        x2 = nn.functional.relu(self.conv1(x1) + self.w1(x1))
        x3 = nn.functional.relu(self.conv2(x2) + self.w2(x2))
        x4 = nn.functional.relu(self.conv3(x3) + self.w3(x3))

        x = x4.permute(0, 2, 3, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ==================== train_FNO 类（完整修复版） ====================

class train_FNO:
    def __init__(self, vtk_dir, resolution=64, modes1=12, modes2=12, width=32, epochs=2000):
        self.vtk_dir = vtk_dir
        self.resolution = resolution
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
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

        # 兼容字段名
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

        # bounding box
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        xi = np.linspace(x_min, x_max, self.resolution)
        yi = np.linspace(y_min, y_max, self.resolution)
        x_grid, y_grid = np.meshgrid(xi, yi, indexing="ij")
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

        # 插值
        ux_g = self._interp_field(points, ux, grid_points)
        uy_g = self._interp_field(points, uy, grid_points)
        p_g = self._interp_field(points, p, grid_points)

        # 坐标归一化到 [-1, 1]
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
                print(nu_val, rho_val, inlet_val)

        '''
        if params_override is not None:
            nu_val = params_override.get("nu", 0.01)
            rho_val = params_override.get("rho", 1.0)
            inlet_val = params_override.get("inlet", 1.0)
        else:
            nu_val, rho_val, inlet_val = 0.01, 1.0, 1.0
        '''


        X_single = np.stack([
            x_grid, y_grid,
            np.full_like(x_grid, nu_val),
            np.full_like(x_grid, rho_val),
            np.full_like(x_grid, inlet_val),
        ], axis=-1).astype(np.float32)

        Y_single = np.stack([ux_g, uy_g, p_g], axis=-1).astype(np.float32)

        return X_single, Y_single

    # ---- 批量构建（无参数） ----
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
                    print(f"  Processed {i+1}/{len(files)}")
            except Exception as e:
                print(f"  Skipped {f}: {e}")

        self.X = np.array(X_list).astype(np.float32)
        self.Y = np.array(Y_list).astype(np.float32)
        print(f"Dataset built: X={self.X.shape}, Y={self.Y.shape}")
        return self.X, self.Y

    # ---- 批量构建（带参数） ----
    def build_dataset_from_params(self, params_dict):
        files = sorted(glob.glob(os.path.join(self.vtk_dir, "case_*.vtk")))
        if len(files) == 0:
            raise FileNotFoundError(f"No case_*.vtk found in {self.vtk_dir}")
        print(f"Found {len(files)} VTK files with params.")

        X_list, Y_list = [], []
        for i, f in enumerate(files):
            case_name = os.path.basename(f).replace(".vtk", "")
            prm = params_dict.get(case_name, None)
            try:
                X_s, Y_s = self.load_single_case(f, params_override=prm)
                X_list.append(X_s)
                Y_list.append(Y_s)
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(files)}")
            except Exception as e:
                print(f"  Skipped {f}: {e}")

        self.X = np.array(X_list).astype(np.float32)
        self.Y = np.array(Y_list).astype(np.float32)
        print(f"Dataset built: X={self.X.shape}, Y={self.Y.shape}")
        return self.X, self.Y

    # ---- 构建模型 ----
    def build_model(self):
        self.model = FNO2d(
            modes1=self.modes1, modes2=self.modes2, width=self.width,
            in_channels=5, out_channels=3,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"FNO model built: {n_params:,} trainable parameters")




        print("X[0, 0, 0, :] =", fno.X[0, 0, 0, :])
        print("X[1, 0, 0, :] =", fno.X[1, 0, 0, :])
        print("Are all X identical?", np.allclose(fno.X[0], fno.X[1]))

        return self.model

    # ---- 训练 ----
    def train(self, batch_size=16, lr=1e-3, val_split=0.2):
        if self.X is None or self.Y is None:
            raise ValueError("Call build_dataset() first!")
        if self.model is None:
            self.build_model()

        # ============================================================
        # ★ 调参旋钮区（只改这里，别动下面）
        # ============================================================
        GRAD_CLIP = 1.0  # 梯度裁剪，0=关，推荐 0.5~2.0
        WEIGHT_P = 1.0  # 压力 loss 权重，p 场太大就降到 0.3~0.5
        WEIGHT_UX = 1.0  # ux 权重
        WEIGHT_UY = 1.0  # uy 权重
        LR = lr  # 学习率，推荐 1e-3 ~ 3e-3
        EARLY_STOP_PATIENCE = 80  # val_loss 多少轮不降就停
        # ============================================================

        # ---- 归一化（X 和 Y 都做） ----
        self.x_mean = np.mean(self.X, axis=(0, 1, 2), keepdims=True)
        self.x_std = np.std(self.X, axis=(0, 1, 2), keepdims=True) + 1e-8
        X_norm = (self.X - self.x_mean) / self.x_std

        self.y_mean = np.mean(self.Y, axis=(0, 1, 2), keepdims=True)
        self.y_std = np.std(self.Y, axis=(0, 1, 2), keepdims=True) + 1e-8
        Y_norm = (self.Y - self.y_mean) / self.y_std

        print(f"X norm range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")
        print(f"Y norm range: [{Y_norm.min():.2f}, {Y_norm.max():.2f}]")
        print(f"Y std per channel (ux, uy, p): {self.y_std.flatten()}")

        # ---- train/val split ----
        N = X_norm.shape[0]
        indices = np.random.permutation(N)
        n_val = int(N * val_split)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_norm[train_idx]).float(),
                          torch.from_numpy(Y_norm[train_idx]).float()),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_norm[val_idx]).float(),
                          torch.from_numpy(Y_norm[val_idx]).float()),
            batch_size=batch_size, shuffle=False
        )

        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=LR)
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

                # ★ 加权求和
                loss = WEIGHT_UX * loss_ux + WEIGHT_UY * loss_uy + WEIGHT_P * loss_p

                loss.backward()

                # ★ 梯度裁剪
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

            # ★ early stopping
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
    def save_model(self, path="fno_ns_model.pth"):
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

    def load_model(self, path="fno_ns_model.pth"):
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

            X_norm = (X_input - self.x_mean) / self.x_std
            X_tensor = torch.from_numpy(X_norm).float().to(self.device)
            y_pred_norm = self.model(X_tensor).cpu().numpy()
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

    # ---- 训练 FNO ----
    print("\n" + "=" * 60)
    print("Training FNO...")
    print("=" * 60)

    fno = train_FNO(vtk_dir=vtk_dir, resolution=64, modes1=6, modes2=6, width=16, epochs=200)
    fno.build_dataset()
    fno.build_model()
    losses = fno.train(batch_size=16, lr=1e-3, val_split=0.2)

    # 保存
    fno.save_model("fno_ns_model.pth")

    # 测试预测（返回物理空间值）
    test_input = fno.X[0]
    pred = fno.predict(test_input)
    print(f"\nPrediction shape: {pred.shape}")
    print(f"Pred ux max: {pred[:,:,0].max():.4f}")
    print(f"True  ux max: {fno.Y[0,:,:,0].max():.4f}")
    print(f"Pred p max: {pred[:,:,2].max():.4f}")
    print(f"True  p max: {fno.Y[0,:,:,2].max():.4f}")

    print("\nALL DONE.")