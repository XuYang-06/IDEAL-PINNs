# IDEAL-PINNs: 基于朗之万动力学与逆密度排斥的自适应进化采样策略

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active_Research-blue)]()

> **I**nverse-**D**ensity **E**volutionary **A**daptive **L**angevin Sampling (**IDEAL**)
> (逆密度朗之万自适应进化采样)

**IDEAL-PINNs** 是一个旨在加速物理信息神经网络 (PINNs) 收敛的新型自适应采样框架。它通过在**探索**（通过逆密度排斥）和**利用**（通过朗之万梯度上升）之间建立**理想平衡**，解决了传统残差采样方法中臭名昭著的“模态坍缩”问题。

---

##  IDEAL (理论框架)

标准的自适应策略（如 RAD 或 RAR）往往难以维持采样分布 $q(x)$ 的**信息熵** $\mathcal{H}(q)$，易受**谱偏差 (Spectral Bias)** 影响。这导致采样点全部聚集在单一的高误差区域，而忽略了其他关键的物理特征。

**IDEAL** 通过构建一个统一的**信息增益**目标，在采样分布 $q(x)$ 和目标分布 $\pi(x)$ 之间建立动态平衡。

### 1. 目标函数与目标分布

PINN 训练的目标是最小化定义域 $\Omega$ 上残差损失的期望：


我们定义一个**目标采样分布** $\pi(x)$，它与 PDE 残差的指数成正比（将残差视为能量）：

```math
\pi(x) \propto \exp(\beta \cdot \mathcal{L}_{r}(x))
```

### 2. 基于有效势能的朗之万动力学

我们采用**随机梯度朗之万动力学 (SGLD)** 来演化粒子。粒子的运动遵循**有效势能** $U_{eff}(x)$ 的梯度：

```math
d x(t)
= - \nabla U_{\mathrm{eff}}(x)\, dt
  + \sqrt{2\eta}\, dW(t)
```

其中，**物理势能**定义为负残差（引导粒子向高 Loss 区域移动）：

```math
U_{\mathrm{phys}}(x) = -\mathcal{L}_{r}(x)
\quad\Longrightarrow\quad
-\nabla U_{\mathrm{phys}}(x)
= \nabla \mathcal{L}_{r}(x)
```

### 3. 逆密度排斥势能

为了防止模态坍缩，我们引入一个基于当前粒子密度 $\rho(x)$ 的**排斥势能**：

```math
U_{\mathrm{rep}}(x)
= \lambda \cdot \log(\rho(x))
```

### 4. IDEAL 联合动态方程

通过叠加物理吸引力和密度排斥力，我们推导出 **IDEAL 随机微分方程 (SDE)**：

```math
dx(t)
= \underbrace{\nabla\mathcal{L}_{r}(x)\, dt}_{\text{物理吸引力}}
  - \underbrace{\lambda \nabla \log(\rho(x))\, dt}_{\text{密度排斥力}}
  + \underbrace{\sqrt{2\eta}\, dW(t)}_{\text{随机探索}}
```

该方程确保粒子的稳态分布能够覆盖所有高残差区域，而非坍缩到单一极值点。。

```math
P(x)
\propto
\frac{\mathcal{L}(x)^{\alpha}}{\rho(x) + \epsilon}
```

---

##  核心特性

*  梯度驱动 (grad Powered)**: 利用每个点的梯度方向作为朗之万方程的飘移项。
*  消除模态坍缩**: 引入排斥性密度项，保证采样器能同时覆盖多个误差峰值，防止“抱团”现象。
*  时间一致性**: 非常适合求解时变 PDE（如 Allen-Cahn, Navier-Stokes），能够自动追踪随时间移动的解特征（如波前）。

---

##  性能表现

| 方法 | 核心机制 | 收敛速度 | 样本多样性 | 计算成本 |
| :--- | :--- | :--- | :--- | :--- |
| **Uniform** (均匀采样) | 随机撒点 | 慢 | 高 | 低 |
| **RAD** (DeepXDE) | 基于 Loss 加权 | 中等 | 低 (易聚集) | 低 |
| **RAR** (Refinement) | 大池子贪心筛选 | 慢 | 中等 | 高 ($10\times$) |
| **IDEAL (Ours)** | **进化 + 排斥** | **极快** | **高 (自适应)** | **中等** |

> ![1D Burgers by IDEAL](images/Burgers1.png)


---

## 安装与使用

###  依赖环境
```bash
pip install numpy torch scipy matplotlib
```

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


## 作者 (Authors)

| [徐阳 (Xu Yang)](https://www.google.com/search?q=https://github.com/XuYang-06) | 陈宏涛 (Chen Hongtao) | 杨佳宁 (Yang Jianing ) | 郭科文 (Guo Kewen) |
| :---: | :---: | :---: | :---: |
| *吉林大学* | *吉林大学* | *吉林大学* | *吉林大学* |
| CV, PINN, 具身智能 | 深度学习, PINN | 深度学习, PINN | 深度学习, PINN |

---
