import torch

class ADMM():
    def __init__(self, rho=5., l1_penalty=0.15, tol=1e-6, max_iter=2000, device="cuda"):
        self.rho = rho
        self.l1_penalty = l1_penalty
        self.tol = tol
        self.max_iter = max_iter
        self.device = device
        self.Q_cho = None

    def precompute(self, C):
        c = C.shape[0]
        Q = 2 * (C @ C.T) + self.rho * torch.eye(c, device=self.device)
        self.Q_cho = torch.linalg.cholesky(Q)

    def step(self, Cb, z, u):
        x = torch.cholesky_solve(
            2 * Cb + self.rho * (z - u),
            self.Q_cho
        )
        z_new = torch.clamp(x + u - self.l1_penalty / self.rho, min=0)
        u_new = u + x - z_new
        return x, z_new, u_new

    def fit(self, C, v):
        if self.Q_cho is None:
            self.precompute(C)

        Cb = C @ v.T
        c = C.shape[0]

        z = torch.randn((c, v.shape[0]), device=self.device)
        u = torch.randn((c, v.shape[0]), device=self.device)

        for _ in range(self.max_iter):
            z_old = z
            x, z, u = self.step(Cb, z, u)

            if (
                torch.linalg.norm(x - z, dim=0).max() < self.tol
                and torch.linalg.norm(self.rho * (z - z_old), dim=0).max() < self.tol
            ):
                break

        return z.T
