# IDEAL-PINNs
针对物理信息神经网络 (PINNs) 在求解高梯度偏微分方程时的采样点“模态坍缩”问题，我们独立提出了一种融合朗之万动力学与逆密度排斥机制的新型采样算法。相比主流的 RAR/RAD 方法，该算法在保证样本多样性的同时，提升了收敛速度与求解精度。We designed a novel sampling algorithm to address the "mode collapse" issue in Physics-Informed Neural Networks (PINNs) when solving high-gradient PDEs. By integrating Langevin Dynamics with an Inverse Density Repulsion mechanism,the method show better algorithm sample diversity and significantly faster convergence compared to SOTA baselines (e.g., RAR, RAD).
# IDEAL-PINNs: 面向物理信息神经网络的“理想”采样策略

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active_Research-blue)]()

> **I**nverse-**D**ensity **E**volutionary **A**daptive **L**angevin Sampling (**IDEAL**)
> (逆密度朗之万自适应进化采样)

**IDEAL-PINNs** 是一个旨在加速物理信息神经网络 (PINNs) 收敛的新型自适应采样框架。它通过在**探索**（通过逆密度排斥）和**利用**（通过朗之万梯度上升）之间建立**理想平衡**，解决了传统残差采样方法中臭名昭著的“模态坍缩”问题。

---

## 为什么选择 IDEAL? (理论框架)

标准的自适应策略（如 RAD 或 RAR）往往难以维持采样分布 $q(x)$ 的**信息熵** $\mathcal{H}(q)$，易受**谱偏差 (Spectral Bias)** 影响。这导致采样点全部聚集在单一的高误差区域，而忽略了其他关键的物理特征。

**IDEAL** 通过构建一个统一的**信息增益**目标，在采样分布 $q(x)$ 和目标分布 $\pi(x)$ 之间建立动态平衡。

### 1. 目标函数与目标分布

PINN 训练的目标是最小化定义域 $\Omega$ 上残差损失的期望：
$$
\mathcal{L}_{PINN} = \mathbb{E}_{x \sim q(x)} [\mathcal{L}_{r}(x)] + \mathcal{L}_{bc} + \mathcal{L}_{ic}
$$

我们定义一个**目标采样分布** $\pi(x)$，它与 PDE 残差的指数成正比（将残差视为能量）：
$$
\pi(x) \propto \exp(\beta \cdot \mathcal{L}_{r}(x))
$$

### 2. 基于有效势能的朗之万动力学

我们采用**随机梯度朗之万动力学 (SGLD)** 来演化粒子。粒子的运动遵循**有效势能** $U_{eff}(x)$ 的梯度：
$$
d x(t) = - \nabla U_{eff}(x) dt + \sqrt{2\eta} d W(t)
$$

其中，**物理势能**定义为负残差（引导粒子向高 Loss 区域移动）：
$$
U_{phys}(x) = - \mathcal{L}_{r}(x) \implies - \nabla U_{phys}(x) = \nabla \mathcal{L}_{r}(x)
$$

### 3. 逆密度排斥势能

为了防止模态坍缩，我们引入一个基于当前粒子密度 $\rho(x)$ 的**排斥势能**：
$$
U_{rep}(x) = \lambda \cdot \log(\rho(x))
$$

### 4. IDEAL 联合动态方程

通过叠加物理吸引力和密度排斥力，我们推导出 **IDEAL 随机微分方程 (SDE)**：
$$
d x(t) = \underbrace{\nabla \mathcal{L}_{r}(x) dt}_{\text{物理吸引力}} - \underbrace{\lambda \nabla \log(\rho(x)) dt}_{\text{密度排斥力}} + \underbrace{\sqrt{2\eta} d W(t)}_{\text{随机探索}}
$$

该方程确保粒子的稳态分布能够覆盖所有高残差区域，而非坍缩到单一极值点。。

$$
\text{最终选择概率: } \quad P(x) \propto \frac{\mathcal{L}(x)^\alpha}{\rho(x) + \epsilon}
$$

---

## 🚀 核心特性

* **⚡ 自动微分驱动 (Auto-grad Powered)**: 利用 PyTorch 的自动微分进行精确、与维度无关的梯度计算（彻底摒弃了低效且有误差的有限差分）。
* **🌊 消除模态坍缩**: 引入排斥性密度项，保证采样器能同时覆盖多个误差峰值，防止“抱团”现象。
* **🧠 时间一致性**: 非常适合求解时变 PDE（如 Allen-Cahn, Navier-Stokes），能够自动追踪随时间移动的解特征（如波前）。
* **📦 开箱即用**: 设计为标准 PINN 数据加载器的直接替代品。

---

## 📊 性能表现

| 方法 | 核心机制 | 收敛速度 | 样本多样性 | 计算成本 |
| :--- | :--- | :--- | :--- | :--- |
| **Uniform** (均匀采样) | 随机撒点 | 慢 | 高 | 低 |
| **RAD** (DeepXDE) | 基于 Loss 加权 | 中等 | 低 (易聚集) | 低 |
| **RAR** (Refinement) | 大池子贪心筛选 | 慢 | 中等 | 高 ($10\times$) |
| **IDEAL (Ours)** | **进化 + 排斥** | **极快** | **高 (自适应)** | **中等** |

> *[图 1: IDEAL 采样点在 1D Burgers 方程上的演化过程。红点代表采样点，完美地追踪并覆盖了激波锋面。]*

---

## 安装与使用

### 1. 依赖环境
```bash
pip install numpy torch scipy matplotlib
```

### 2. 快速上手
只需导入 `IDEALSampler` 并在你的训练循环中使用即可。

```python
import torch
from ideal_sampler import IDEALSampler  # 你的核心文件

# --- 1. 定义 PDE 残差 (Loss 函数) ---
def pde_residual(xy):
    """
    xy: [Batch, Dimension] tensor, requires_grad=True
    返回: [Batch, 1] 残差的平方
    """
    # 示例: u_t + u*u_x = \nu * u_xx
    u = model(xy)
    # ... 计算梯度和物理残差 ...
    residual = ... 
    return residual ** 2

# --- 2. 初始化 IDEAL 采样器 ---
sampler = IDEALSampler(
    loss_fn=pde_residual,
    domain_bounds=[[-1.0, 1.0], [0.0, 1.0]], # 时空定义域
    device='cuda',
    use_autograd=True  # 开启自动微分以获得高精度与扩展性
)

# --- 3. 训练循环 ---
for epoch in range(max_epochs):
    # 获取自适应采样点 (进化过程在内部自动完成)
    # 采样器会自动管理种群记忆
    x_train = sampler.get_samples(n_samples=2000)
    
    # 标准的 PINN 更新步骤
    loss = train_step(x_train)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.5f}")
```

---

## 开发路线图 (Roadmap)

- **核心算法**: 实现基于网格密度估计的 IDEAL 采样。
- **性能优化**: 实现基于自动微分 (Auto-grad) 的朗之万动力学。
- **高维扩展**: 实现基于 KNN 的无网格密度估计，以支持高维 (>3D) PDE。
- **基准测试**: 在 Navier-Stokes 方程上与 SVGD 和 Failure-Informed Sampling 进行全面对比。

---

## 引用 (Citation)

如果您觉得 **IDEAL-PINNs** 对您的研究有帮助，请考虑引用：

```bibtex
@misc{ideal_pinn_2025,
  author = {Xu Yang},
  title = {IDEAL-PINNs: Inverse-Density Evolutionary Adaptive Langevin Sampling},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/XuYang-06/IDEAL-PINNs](https://github.com/XuYang-06/IDEAL-PINNs)}}
}
```

---


## 👤 作者 (Authors)

| [徐阳 (Xu Yang)](https://www.google.com/search?q=https://github.com/XuYang-06) | 陈宏涛 (Hongtao Chen) | 杨佳宁 (Jianing Yang) | 郭科文 (Kewen Guo) |
| :---: | :---: | :---: | :---: |
| *吉林大学* | *吉林大学* | *吉林大学* | *吉林大学* |
| CV, PINN, 具身智能 | 深度学习, PINN | 深度学习, PINN | 深度学习, PINN |

---
