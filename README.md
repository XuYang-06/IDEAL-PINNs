# IDEAL-PINNs: 面向物理信息神经网络的“理想”采样策略

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active_Research-blue)]()

> **I**nverse-**D**ensity **E**volutionary **A**daptive **L**angevin Sampling (**IDEAL**)
> (逆密度朗之万自适应进化采样)

**IDEAL-PINNs** 是一个旨在加速物理信息神经网络 (PINNs) 收敛的新型自适应采样框架。它通过在**探索**（通过逆密度排斥）和**利用**（通过朗之万梯度上升）之间建立**理想平衡**，解决了传统残差采样方法中臭名昭著的“模态坍缩”问题。

---

##  IDEAL 理论框架


标准的自适应策略（如 RAD 或 RAR）往往难以维持采样分布 $q(x)$ 的**信息熵** $\mathcal{H}(q)$，极易受**谱偏差 (Spectral Bias)** 影响。这会导致采样点全部挤在单一的高误差模态附近，而忽略了其他同样重要的物理特征（例如移动的波前、弱奇点等）。

从信息论角度看，我们真正想要的是一个采样分布 $q(x)$，既要**偏向高残差区域**，又要**保持足够的全局多样性**。  
一个非严格但直观的目标可以写成：

```math
\max_{q} \ \mathbb{E}_{x\sim q}[\mathcal{L}_r(x)]
\;+\;
\lambda \mathcal{H}(q),
```

其中：

```math
\qquad
\mathcal{H}(q) = -\int q(x)\log q(x)\,dx,
````

其中：

* 第一项鼓励 $q$ 聚焦在**高残差**的“难点”区域（Exploitation）。
* 第二项通过信息熵 $\mathcal{H}(q)$ 惩罚过度集中的分布，鼓励采样点**铺开来**（Exploration）。

**IDEAL** 正是一个用**粒子动力学 + 逆密度排斥**来近似实现上述熵正则目标的统一框架。

---

## 1. 目标采样分布与能量视角

PINN 训练的目标是最小化定义域 $\Omega$ 上残差损失的期望：

```math
\min_{\theta} \ \mathbb{E}_{x\sim q(x)}\big[\mathcal{L}_{r}(x; \theta)\big],
```

其中 $\mathcal{L}_r(x; \theta)$ 表示在点 $x$ 处的 PDE 残差（或残差相关的标量度量）。

我们定义一个**目标采样分布** $\pi(x)$，将残差视为“能量”并通过 Boltzmann 形式加权：

```math
\pi(x) \propto \exp\big(\beta \cdot \mathcal{L}_{r}(x)\big),
```

其中 $\beta>0$ 可理解为一个“温度系数”，控制对高残差区域的敏感程度。
在理想情形下，我们希望 $q(x)$ 能够接近该目标分布 $\pi(x)$，但同时又不发生模式坍缩。

---

## 2. 基于有效势能的朗之万动力学 (Langevin Dynamics)

为在连续空间中演化采样点，我们采用**随机梯度朗之万动力学 (SGLD)**。
粒子的运动被视为在一个**有效势能** $U_{\mathrm{eff}}(x)$ 下的随机动力学过程：

```math
d x(t)
= - \nabla U_{\mathrm{eff}}(x)\, dt
  + \sqrt{2\eta}\, dW(t),
```

其中 $W(t)$ 是标准 Wiener 过程，$\eta>0$ 控制噪声强度。

### 2.1 物理势能：朝高残差区域“吸引”

我们首先定义**物理势能**：

```math
U_{\mathrm{phys}}(x) = -\mathcal{L}_{r}(x),
\quad\Longrightarrow\quad
-\nabla U_{\mathrm{phys}}(x)
= \nabla \mathcal{L}_{r}(x).
```

这意味着：

* 当残差 $\mathcal{L}_r(x)$ 较大时，对应的势能更“低”，粒子更倾向于沿梯度方向向这些区域移动；
* 这相当于一种**残差驱动的难例挖掘 (Residual-based Hard Example Mining)**。

### 2.2 逆密度排斥势能：

为避免所有粒子都“挤”到同一个高误差模态附近，我们引入基于当前粒子分布 $q(x)$ 的**排斥势能** $U_{\mathrm{rep}}(x)$。
设 $\rho(x)$ 表示粒子的局部密度估计，则定义：

```math
U_{\mathrm{rep}}(x)
= \lambda \cdot \log(\rho(x)),
```

其中 $\lambda>0$ 控制排斥强度。

从信息论角度看，$\log \rho(x)$ 与分布熵 $\mathcal{H}(q) = -\int q(x)\log q(x)\,dx$ 密切相关。

在势能中加入 $U_{\mathrm{rep}}(x) = \lambda \log \rho(x)$ 等价于对采样分布施加一种“熵正则化”（entropy regularization）

- **高密度区域**： $\rho(x)$  大， $\log\rho(x)$  大， $U_{\mathrm{rep}}(x)$ 被放大；
- **低密度区域**： $\rho(x)$  小， $U_{\mathrm{rep}}(x)$  较小。

因此， $U_{\mathrm{rep}}$  实际上在**显式鼓励高熵、高多样性**的采样分布。


---

## 3. IDEAL 联合有效势能与动力学

我们将上述两种势能叠加，构造出 IDEAL 的联合有效势能：

```math
U_{\mathrm{IDEAL}}(x)
= U_{\mathrm{phys}}(x) + U_{\mathrm{rep}}(x)
= -\mathcal{L}_r(x) + \lambda \log\rho(x).
```

对应的粒子动态方程为：

```math
dx(t)
= \underbrace{\nabla\mathcal{L}_{r}(x)\, dt}_{\text{物理吸引力 (Exploitation)}}
  - \underbrace{\lambda \nabla \log(\rho(x))\, dt}_{\text{密度排斥力 / 高熵驱动 (Exploration)}}
  + \underbrace{\sqrt{2\eta}\, dW(t)}_{\text{随机探索 (Stochastic Exploration)}}.
```

**直观理解：**

* 第一项：沿残差梯度上升，粒子自动靠近**高误差区域**；
* 第二项：沿 $-\nabla\log\rho(x)$ 方向移动，**从高密度“拥挤区”被推开**，提高采样的多样性和信息熵；
* 第三项：高斯噪声提供额外的随机探索能力。

在很多已有的朗之万采样与变分推断工作中，可以证明类似的动力系统在温度 $\eta$ 固定的情况下，其稳态分布 $q^*(x)$ 与 $\exp(-U_{\mathrm{IDEAL}}(x)/\eta)$ 成正比。
本工作不追求严格证明，但这一联系为 IDEAL “**兼顾高残差与高熵**” 的行为提供了直观解释。

---

## 4. 自适应高熵选择机制

在实际实现中，我们不会显式求解稳态分布，而是通过构造一个**概率选择场**来在“利用 (Exploitation)” 与 “探索 (Exploration)”之间维持动态平衡，从而解决 **PINNs** 训练中的模态坍缩问题。

我们定义采样概率 $P(x)$ 如下：

```math
P(x) \propto
\underbrace{\mathcal{L}(x)^{\alpha}}_{\text{残差吸引 (利用)}}
\cdot
\underbrace{\frac{1}{\hat{\rho}_{\text{KNN}}(x) + \epsilon}}_{\text{逆密度排斥 (高熵探索)}},
```

其中：

* $\mathcal{L}(x)$：PDE 在点 $x$ 处的残差（或残差相关 loss）；
* $\alpha>0$：聚焦系数（通常取 $1.5 \sim 2.0$），控制对高误差区域的敏感度；
* $\hat{\rho}_{\text{KNN}}(x)$：基于 K 近邻 (KNN) 的局部密度估计。

该概率场由以下两个关键分量共同驱动：

### 4.1 残差驱动 (Residual Exploitation): $\mathcal{L}(x)^{\alpha}$

* **机制**：残差越大，采样概率越高；
* **物理含义**：实现了类似 **Hard Example Mining** 的效果，算法会自动赋予激波锋面、奇点或边界层等高误差区域更高权重。

### 4.2 基于 KNN 的逆密度排斥 (KNN-based Repulsion): $(\hat{\rho} + \epsilon)^{-1}$

为保证样本多样性（即**采样熵高**），我们利用 **k-近邻 (k-Nearest Neighbors)** 构建一个**无网格 (Mesh-free)** 的局部密度估计量：

```math
\hat{\rho}_{\text{KNN}}(x)
\propto
\left(
  \frac{1}{k} \sum_{x_j \in \mathcal{N}_k(x)} \|x - x_j\|_2
\right)^{-1}.
```

* **无网格特性**：不依赖空间网格划分，天然适应**高维 PDE** 问题，缓解维数灾难；
* **排斥力 / 高熵效应**：

  * 当粒子在某处过度聚集时，其最近邻平均距离减小，导致估计密度 $\hat{\rho}$ 升高；
  * 于是 $(\hat{\rho}+\epsilon)^{-1}$ 急剧减小，采样概率 $P(x)$ 降低；
  * 这会产生一种“**排斥力**”，迫使粒子群向低密度的稀疏区域扩散，从而在训练过程中维持更高的**信息熵**，有效抑制 **mode collapse**。

---

## 5. 核心特性 (Key Features)

* **高熵自适应采样 (High-Entropy Adaptive Sampling)**：
  显式引入逆密度项，通过 $\log\rho(x)$ 和 $(\hat\rho+\epsilon)^{-1}$ 机制在训练过程中持续维持**较高的信息熵**，防止采样点“抱团”。
* **自动微分驱动 (Auto-grad Powered)**：
  利用 PyTorch 自动微分进行精确、与维度无关的残差与梯度计算，摒弃有限差分。
* **消除模态坍缩 (Mode-Collapse Mitigation)**：
  物理吸引 + 逆密度排斥保证采样器既能锁定多个误差峰值，又能保持全局多样性。
* **时间一致性 (Temporal Consistency)**：
  非常适合时变 PDE（如 Allen-Cahn, Navier-Stokes），能够自动追踪随时间移动的解特征（如波前、界面）。

---

## 6. 性能表现 (Performance Overview)

| 方法                   | 核心机制                 | 收敛速度   | 样本多样性 / 熵    | 计算成本           |
| :------------------- | :------------------- | :----- | :----------- | :------------- |
| **Uniform** (均匀采样)   | 随机撒点                 | 慢      | 高（但不聚焦）      | 低              |
| **RAD** (DeepXDE)    | 基于 Loss 加权           | 中等     | 低（易聚集）       | 低              |
| **RAR** (Refinement) | 大池子贪心筛选              | 慢      | 中等           | 高 ($10\times$) |
| **IDEAL (Ours)**     | **Langevin + 逆密度排斥** | **极快** | **高 (显式高熵)** | **中等**         |

> ![1D Burgers by IDEAL](images/Burgers1.png)
> *图：IDEAL 在 1D Burgers 方程上自动将采样点集中在激波附近，同时保持全局覆盖。*

---

## 7. 安装与使用

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


## 8. 作者 (Authors)

| [徐阳 (Xu Yang)](https://www.google.com/search?q=https://github.com/XuYang-06) | 陈宏涛 (Chen Hongtao) | 杨佳宁 (Yang Jianing ) | 郭科文 (Guo Kewen) |
| :---: | :---: | :---: | :---: |
| *吉林大学* | *吉林大学* | *吉林大学* | *吉林大学* |
| CV, PINN, 具身智能 | 深度学习, PINN | 深度学习, PINN | 深度学习, PINN |

---
