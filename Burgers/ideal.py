import numpy as np
import torch

class IDEALSampler:
   

    def __init__(self, loss_fn, bounds, device=None, 
                 grid_res=50, alpha=4.0, noise_std=0.03, 
                 step_size=0.02, uniform_ratio=0.05, fd_epsilon=0.01):
       
        self.loss_fn = loss_fn
        self.device = device
        
       
        self.bounds = np.array(bounds, dtype=np.float32)
        self.dim = self.bounds.shape[0]
        
      
        self.grid_res = grid_res
        self.alpha = alpha
        self.noise_std = noise_std
        self.step_size = step_size
        self.uniform_ratio = uniform_ratio
        self.epsilon = fd_epsilon
        
       
        self.particles = None

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _process_loss(self, xy):
        try:
            
            with torch.no_grad():
                xy_t = self._to_tensor(xy)
                val = self.loss_fn(xy_t)
                
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            
           
            raw_loss = np.array(val, dtype=np.float32).reshape(-1)
        except Exception as e:
            print(f"[IDEAL Warning] Loss calculation failed: {e}")
            raw_loss = np.zeros(xy.shape[0], dtype=np.float32)
        return raw_loss

    def _compute_gradients_finite_diff(self, xy):
       
        grads = np.zeros_like(xy)
        N, D = xy.shape
        eps = self.epsilon

       
        for d in range(D):
            
            xy_pos = xy.copy()
            xy_pos[:, d] += eps
            loss_pos = self._process_loss(xy_pos)
     
            xy_neg = xy.copy()
            xy_neg[:, d] -= eps
            loss_neg = self._process_loss(xy_neg)
            
            grads[:, d] = (loss_pos - loss_neg) / (2 * eps)

       
        norm = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-9
        
        mask = (norm > 1e-5).flatten()
        grads_norm = np.zeros_like(grads)
        grads_norm[mask] = grads[mask] / norm[mask]
        
        return grads_norm.astype(np.float32)

    def _compute_weights(self, xy, raw_loss):
      
        N = len(xy)
        
        indices = []
        for d in range(self.dim):
            min_val, max_val = self.bounds[d]
            norm_coord = (xy[:, d] - min_val) / (max_val - min_val + 1e-9)
            idx = (norm_coord * self.grid_res).astype(int)
            idx = np.clip(idx, 0, self.grid_res - 1)
            indices.append(idx)
        
        
        grid_shape = tuple([self.grid_res] * self.dim)
        grid_counts = np.zeros(grid_shape)
       
        np.add.at(grid_counts, tuple(indices), 1)
        
        local_density = grid_counts[tuple(indices)]
        
        log_loss = np.log10(raw_loss + 1e-12)
        min_l, max_l = np.min(log_loss), np.max(log_loss)
        
        if max_l - min_l < 1e-6:
            norm_loss = np.zeros_like(log_loss)
        else:
            norm_loss = (log_loss - min_l) / (max_l - min_l)
            
    
        weights = (norm_loss ** self.alpha) / (local_density + 1.0)
        
        
        sum_w = np.sum(weights)
        if sum_w == 0:
            return np.ones(N) / N
        return weights / sum_w

    def sample(self, n_samples, init_points=None, generations=1):
       
        if self.particles is None or len(self.particles) != n_samples:
            if init_points is not None:
                current = np.array(init_points, dtype=np.float32)
            else:
                current = np.random.rand(n_samples, self.dim).astype(np.float32)
                for d in range(self.dim):
                    min_v, max_v = self.bounds[d]
                    current[:, d] = current[:, d] * (max_v - min_v) + min_v
        else:
            current = self.particles.copy()

        n_uniform = int(n_samples * self.uniform_ratio)
        n_resample = n_samples - n_uniform
        
        for _ in range(generations):
            raw_loss = self._process_loss(current)
            
            probs = self._compute_weights(current, raw_loss)
            indices = np.random.choice(len(current), size=n_resample, p=probs)
            seeds = current[indices]
            
           
            noise = np.random.randn(n_resample, self.dim) * self.noise_std
            xy_bloom = seeds + noise
            
           
            grads = self._compute_gradients_finite_diff(xy_bloom)
            xy_final = xy_bloom + grads * self.step_size
            
           
            for d in range(self.dim):
                min_v, max_v = self.bounds[d]
                xy_final[:, d] = np.clip(xy_final[:, d], min_v, max_v)
            
            
            xy_uniform = np.random.rand(n_uniform, self.dim).astype(np.float32)
            for d in range(self.dim):
                min_v, max_v = self.bounds[d]
                xy_uniform[:, d] = xy_uniform[:, d] * (max_v - min_v) + min_v
            
            
            current = np.concatenate([xy_final, xy_uniform], axis=0)

       
        self.particles = current
        
        return current