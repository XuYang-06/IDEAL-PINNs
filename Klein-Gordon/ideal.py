import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import contextlib

class IDEALSampler:
 

    def __init__(self, loss_fn, bounds, device=None, 
                 k=80, alpha=4.0, noise_std=0.05, 
                 step_size=0.01, uniform_ratio=0.2, 
              
                 grid_res=None, fd_epsilon=None):

        self.loss_fn = loss_fn
        self.device = device
        self.bounds = torch.tensor(bounds, dtype=torch.float32, device=device)
        self.dim = self.bounds.shape[0]
        
        # IDEAL 参数
        self.k = k
        self.alpha = alpha
        self.noise_std = noise_std
        self.step_size = step_size
        self.uniform_ratio = uniform_ratio
        
        self.particles = None

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _process_loss(self, xy):
        """仅用于计算权重的 Loss (不需要梯度)"""
        try:
            xy_t = self._to_tensor(xy)
            with torch.no_grad():
                val = self.loss_fn(xy_t)
            
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            
            raw_loss = np.array(val, dtype=np.float32).reshape(-1)
        except Exception as e:
            # 容错：如果 loss_fn 内部必须要有梯度才能运行，这里可能会捕获异常
            print(f"[IDEAL Warning] Weight calculation error: {e}")
            raw_loss = np.zeros(len(xy), dtype=np.float32)
        return raw_loss

    def _compute_gradients_autograd(self, xy_np):
        """
        [核心] 自动微分计算梯度
        """
        # 1. 准备 Tensor，开启梯度
        xy_tensor = self._to_tensor(xy_np).requires_grad_(True)
        
        # 2. 强制开启梯度上下文 (解决你遇到的报错)
        # 即使外部环境是 no_grad，这里也强制开启
        with torch.enable_grad():
            # CUDA 保护
            ctx = torch.cuda.device(self.device) if 'cuda' in str(self.device) else contextlib.nullcontext()
            
            with ctx:
                # 3. 计算 Loss
                loss = self.loss_fn(xy_tensor)
                
                # 检查点：如果 loss 没有 grad_fn，说明 loss_fn 内部断开了
                if isinstance(loss, torch.Tensor) and not loss.requires_grad:
                    # 尝试补救：如果是 Tensor 但没梯度，可能是被 detach 了
                    loss.requires_grad_(True) 
                
                # 4. 反向传播
                # 对 loss.sum() 求导，等价于对每个点独立求导
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=xy_tensor,
                    grad_outputs=torch.ones_like(loss),
                    create_graph=False, # 不需要二阶导
                    allow_unused=True   # 防止炸裂
                )[0]
                
                if grads is None:
                    return np.zeros_like(xy_np)

                # 5. 归一化
                norm = torch.norm(grads, dim=1, keepdim=True) + 1e-9
                normalized_grads = grads / norm
            
        return normalized_grads.detach().cpu().numpy()

    def _compute_weights(self, xy, raw_loss):
        """KNN 密度加权"""
        # 1. KNN 距离 (距离大 = 稀疏)
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(xy)
        distances, _ = nbrs.kneighbors(xy)
        sparsity = distances.mean(axis=1) + 1e-9

        # 2. Loss 归一化
        log_loss = np.log10(raw_loss + 1e-12)
        min_l, max_l = np.min(log_loss), np.max(log_loss)
        if max_l - min_l < 1e-6:
            norm_loss = np.zeros_like(log_loss)
        else:
            norm_loss = (log_loss - min_l) / (max_l - min_l)

        # 3. 权重 (Loss * Sparsity = 激波聚焦)
        weights = (norm_loss ** self.alpha) * sparsity

        sum_w = np.sum(weights)
        if sum_w == 0:
            return np.ones(len(xy)) / len(xy)
        return weights / sum_w

    def sample(self, n_samples, init_points=None, generations=1):
        if self.particles is None or len(self.particles) != n_samples:
            if init_points is not None:
                current = np.array(init_points, dtype=np.float32)
            else:
                low = self.bounds[:, 0].cpu().numpy()
                high = self.bounds[:, 1].cpu().numpy()
                current = np.random.rand(n_samples, self.dim).astype(np.float32)
                current = current * (high - low) + low
        else:
            current = self.particles.copy()

        n_uniform = int(n_samples * self.uniform_ratio)
        n_resample = n_samples - n_uniform
        
        low = self.bounds[:, 0].cpu().numpy()
        high = self.bounds[:, 1].cpu().numpy()

        for _ in range(generations):
            # A. 权重
            raw_loss = self._process_loss(current)
            probs = self._compute_weights(current, raw_loss)
            
            # B. 选择
            indices = np.random.choice(len(current), size=n_resample, p=probs)
            seeds = current[indices]
            
            # C. 扩散
            noise = np.random.randn(n_resample, self.dim) * self.noise_std
            xy_bloom = seeds + noise
            
            # D. 修正 (AutoGrad)
            grads = self._compute_gradients_autograd(xy_bloom)
            xy_final = xy_bloom + grads * self.step_size
            
            # 边界
            for d in range(self.dim):
                xy_final[:, d] = np.clip(xy_final[:, d], low[d], high[d])
            
            # E. 注入
            xy_uniform = np.random.rand(n_uniform, self.dim).astype(np.float32)
            xy_uniform = xy_uniform * (high - low) + low
            
            current = np.concatenate([xy_final, xy_uniform], axis=0)

        self.particles = current
        return current