"""LSTM Autoencoder on the raw window [T, C]. Requires PyTorch (optional).

Score = mean squared reconstruction error; per-channel error is returned
for cheap channel-level attribution.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _AE(nn.Module):
        def __init__(self, n_ch: int, hidden: int, latent: int):
            super().__init__()
            self.enc = nn.LSTM(n_ch, hidden, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.from_latent = nn.Linear(latent, hidden)
            self.dec = nn.LSTM(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_ch)

        def forward(self, x):
            _, (h, _) = self.enc(x)
            z = self.to_latent(h[-1])
            h0 = torch.tanh(self.from_latent(z))
            dec_in = h0.unsqueeze(1).repeat(1, x.size(1), 1)
            y, _ = self.dec(dec_in)
            return self.out(y)


class LSTMAutoencoder:
    name = "lstm_ae"

    def __init__(self, n_ch=5, hidden=32, latent=16, epochs=15, lr=1e-3, batch=64, downsample=10, device=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMAutoencoder (pip install torch)")
        self.hidden, self.latent, self.epochs, self.lr, self.batch = hidden, latent, epochs, lr, batch
        self.downsample = downsample  # 1000 samples -> 100 steps keeps training fast
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _AE(n_ch, hidden, latent).to(self.device)
        self.mu = np.zeros(n_ch, dtype=np.float32)
        self.sd = np.ones(n_ch, dtype=np.float32)

    def _prep(self, X: np.ndarray) -> torch.Tensor:
        X = X[:, :: self.downsample, :]
        X = (X - self.mu) / self.sd
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def fit(self, X: np.ndarray) -> "LSTMAutoencoder":
        self.mu = X.mean(axis=(0, 1)).astype(np.float32)
        self.sd = (X.std(axis=(0, 1)) + 1e-6).astype(np.float32)
        Xt = self._prep(X)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for i in range(0, len(Xt), self.batch):
                xb = Xt[perm[i:i + self.batch]]
                opt.zero_grad()
                loss = loss_fn(self.net(xb), xb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            print(f"  [lstm_ae] epoch {ep + 1}/{self.epochs}  loss={tot / len(Xt):.5f}")
        self.net.eval()
        return self

    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def channel_error(self, X: np.ndarray) -> np.ndarray:
        Xt = self._prep(X)
        rec = self.net(Xt)
        return ((rec - Xt) ** 2).mean(dim=1).cpu().numpy()  # [N, C]

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        return self.channel_error(X).mean(axis=1)

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd,
                    "cfg": dict(hidden=self.hidden, latent=self.latent, downsample=self.downsample)}, path)

    @staticmethod
    def load(path) -> "LSTMAutoencoder":
        ck = torch.load(path, map_location="cpu")
        m = LSTMAutoencoder(hidden=ck["cfg"]["hidden"], latent=ck["cfg"]["latent"], downsample=ck["cfg"]["downsample"])
        m.net.load_state_dict(ck["state"])
        m.mu, m.sd = ck["mu"], ck["sd"]
        m.net.eval()
        return m
