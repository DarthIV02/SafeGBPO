import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from safeguards.solver.fsnet_distance import FSNetDistance


def train_fsnet(
    dataset_path: str = "fsnet_dist_dataset.pt",
    out_ckpt: str = "fsnet_dist_ckpt.pt",
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    data = torch.load(dataset_path)
    X = data["x"].float()
    Y = data["y"].float()

    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    N = X.shape[0]
    in_dim = X.shape[1]


    n_train = int(0.9 * N)
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]

    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FSNetDistance(in_dim=in_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"[FSNet] dataset: {dataset_path}, N={N}, in_dim={in_dim}")

    best_val = float("inf")
    best_state = None

    train_hist: list[float] = []
    val_hist: list[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        total_train = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_train += loss.item() * xb.size(0)
        total_train /= n_train

        # ---- val ----
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                total_val += loss.item() * xb.size(0)
        total_val /= (N - n_train)

        train_hist.append(total_train)
        val_hist.append(total_val)

        print(f"[FSNet][epoch {ep:03d}] train={total_train:.4e}  val={total_val:.4e}")

        if total_val < best_val:
            best_val = total_val
            best_state = {
                "model": model.state_dict(),
                "in_dim": in_dim,
            }

    out_path = Path(out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    print(f"[FSNet] Saved best model to {out_path} (val MSE={best_val:.4e})")

    epochs_axis = list(range(1, epochs + 1))

    plt.figure()
    plt.plot(epochs_axis, train_hist, label="train MSE")
    plt.plot(epochs_axis, val_hist, label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.yscale("log")  # 好みで。線形にしたければこの行を消す
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"FSNetDistance training (N={N}, train={n_train}, val={N-n_train})")

    curve_path = Path(out_ckpt).with_suffix(".png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    print(f"[FSNet] Saved loss curve to {curve_path}")

    hist_path = Path(out_ckpt).with_suffix(".hist.pt")
    torch.save(
        {"train": torch.tensor(train_hist), "val": torch.tensor(val_hist)},
        hist_path,
    )
    print(f"[FSNet] Saved loss history to {hist_path}")

if __name__ == "__main__":
    train_fsnet(
        dataset_path="fsnet_dist_dataset.pt",
        out_ckpt="fsnet_dist_ckpt.pt",
        batch_size=256,
        epochs=50,
    )