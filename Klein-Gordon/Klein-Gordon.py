import os
import time
import numpy as np
from pinn import *
from grad_stats import *
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR
from scipy.interpolate import griddata
from pyDOE import lhs
import scipy.io
import random
from collections import deque

# [修改 1] 导入 IDEAL 发行版
from ideal import IDEALSampler
from pinn import *
from grad_stats import *


def plot_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history):
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, label='Total Loss')
    plt.semilogy(loss_pde_history, label='PDE Loss')
    plt.semilogy(loss_bc_history, label='BC Loss')
    plt.semilogy(loss_ic_history, label='IC Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    plt.grid()
    plt.savefig('loss_history.png')
    plt.close()


def save_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history, filename='loss_history.npz'):
    np.savez(filename,
             loss_history=np.array(loss_history),
             loss_pde_history=np.array(loss_pde_history),
             loss_bc_history=np.array(loss_bc_history),
             loss_ic_history=np.array(loss_ic_history))


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.set_printoptions(threshold=np.inf)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练超参数
lr = 1e-3
Adam_n_epochs = 20001
num_0 = 1000     # “IC”点数（第二维的下边界）
num_b = 1000      # 边界点数（左右边界）
num_r = 10000    # PDE 内部点
dim = 2
num_hidden = 50
num_layers = 7
layer_sizes = [dim] + [num_hidden] * num_layers + [1]

N_change_point = 50     # 每隔多少 epoch 重新采样 PDE 点
N_change_lambda = 10    # 每隔多少 epoch 更新 loss 权重


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


set_seed(204)


def compute_L2_error(y_true, y_pred):
    """
    计算相对 L2 误差：对点上相对误差做 RMS
    """
    error = y_true - y_pred
    eps = 1e-12
    denom = np.abs(y_true) + eps
    rel = np.abs(error) / denom
    L2 = np.sqrt(np.mean(rel ** 2))
    return L2, np.max(rel), np.min(rel)


# ===========================
#       主训练函数
# ===========================
def train(bounds, method=1, guding_lr=False):
    """
    bounds: [[lx, rx], [ly, ry]]
    method: 0,1,2 分别对应不同的权重更新方式（保持原脚本约定）
    guding_lr: 是否固定学习率
    """
    # -------- 定义域 --------
    lx, rx = bounds[0]
    ly, ry = bounds[1]

    print(f"Using domain: x in [{lx}, {rx}], t in [{ly}, {ry}]")
    print("Network layers:", layer_sizes)

    # -------- KG 方程参数（与第一个文件保持一致）--------
    # PDE: u_tt - u_xx + u^3 = f(x,t)
    # f(x,t) = u_tt_exact - u_xx_exact + u_exact^3  (默认 alpha=-1, beta=0, gamma=1, k=3)
    alpha = -1.0
    beta = 0.0
    gamma = 1.0
    k_power = 3.0

    # ========= 真解 & 力项定义 =========

    # numpy 版真解，用来生成 u_sol、画图、L2 误差等
    def u_exact_np(X):
        """
        X: numpy array, shape (N, 2), X[:,0]=x, X[:,1]=t
        u(x,t) = x*cos(5π t) + (x*t)^3  —— 来自第一个文件的 kg_equation / u_func
        """
        x = X[:, 0:1]
        t = X[:, 1:2]
        return x * np.cos(5 * np.pi * t) + (x * t) ** 3

    # torch 版真解及导数，用于构造右端项 f(x,t)
    def u_exact_torch(xy):
        x = xy[:, 0:1]
        t = xy[:, 1:2]
        return x * torch.cos(5 * math.pi * t) + (x * t) ** 3

    def u_tt_exact_torch(xy):
        """
        u_tt(x,t) = -25π^2 x cos(5π t) + 6 t x^3
        来自第一个文件的 u_tt 定义
        """
        x = xy[:, 0:1]
        t = xy[:, 1:2]
        return -25 * math.pi ** 2 * x * torch.cos(5 * math.pi * t) + 6 * t * x ** 3

    def u_xx_exact_torch(xy):
        """
        u_xx(x,t) = 6 x t^3
        来自第一个文件的 u_xx 定义
        """
        x = xy[:, 0:1]
        t = xy[:, 1:2]
        return 6 * x * t ** 3

    def f_torch(xy):
        """
        f(x,t) = u_tt + alpha*u_xx + beta*u + gamma*u^k
        默认 alpha=-1, beta=0, gamma=1, k=3 与第一个文件保持一致
        """
        u_ex = u_exact_torch(xy)
        u_tt_ex = u_tt_exact_torch(xy)
        u_xx_ex = u_xx_exact_torch(xy)
        return u_tt_ex + alpha * u_xx_ex + beta * u_ex + gamma * (u_ex ** k_power)

    # ========= 生成真解网格 =========
    x_exact = np.linspace(lx, rx, 1001)[:, None]
    t_exact = np.linspace(ly, ry, 1001)[:, None]
    xx, tt = np.meshgrid(x_exact, t_exact)
    X_grid = np.hstack((xx.flatten()[:, None], tt.flatten()[:, None]))
    u_sol = u_exact_np(X_grid).reshape(xx.shape)

    # -------- PINN 模型 --------
    model = PINN(sizes=layer_sizes, activation=torch.nn.Tanh()).to(device)

    # ========= PDE 残差：KG 算子 =========
    def pde_loss(uhat, data):
        """
        KG 算子作用在网络解 uhat 上：
            KG(u) = u_tt - u_xx + u^3
        我们希望：KG(u) = f(x,t)
        所以 residual = KG(uhat) - f(x,t)
        """
        # 一阶导
        du = grad(outputs=uhat,
                  inputs=data,
                  grad_outputs=torch.ones_like(uhat),
                  create_graph=True)[0]
        dudx = du[:, 0:1]
        dudt = du[:, 1:2]

        # 二阶导
        dudxx = grad(outputs=dudx,
                     inputs=data,
                     grad_outputs=torch.ones_like(dudx),
                     create_graph=True)[0][:, 0:1]
        dudtt = grad(outputs=dudt,
                     inputs=data,
                     grad_outputs=torch.ones_like(dudt),
                     create_graph=True)[0][:, 1:2]

        # KG(u)
        kg_u = dudtt - dudxx + uhat ** 3

        # 右端项 f(x,t)，用真解解析表达式构造
        f_val = f_torch(data)

        residual = kg_u - f_val
        return residual

    # [修改 2] sampler 用的 loss：强制开启梯度
    def loss_fn_for_sampler(xy):
        # IDEALSampler 为了省显存可能会用 no_grad 上下文
        # 但 PDE 残差计算需要求导，所以必须在这里强制开启梯度
        with torch.set_grad_enabled(True):
            if not xy.requires_grad:
                xy.requires_grad_(True)
            uhat = model(xy)
            res = pde_loss(uhat, xy)
            return (res ** 2).flatten() # 确保是一维的

    # -------- [修改 3] IDEALSampler 实例 --------
    Samples = IDEALSampler(
        loss_fn=loss_fn_for_sampler,
        bounds=bounds,
        device=device,
        # 这里使用你实验效果最好的参数配置
        alpha=4.0,       # 保持你原始代码的激进选择
        noise_std=0.03,  # 扩散噪声
        step_size=0.02   # 梯度爬坡步长
    )

    # ========= 采样边界 & PDE 点 =========
    def get_samplers(Samples, N_f=10000, N_ic=100, N_bc=100, T=20):
    
        # 内部 PDE 点初始均匀分布
        init_f = np.random.uniform([lx, ly], [rx, ry], (N_f, 2))

        # 调用 IDEAL 采样
        data_np = Samples.sample(n_samples=N_f, init_points=init_f, generations=T)
        data = torch.tensor(data_np, dtype=torch.float32, device=device)
        data.requires_grad_(True)

        # “IC”：t = ly 的一条线（x 从 lx 到 rx）
        x_ic = torch.linspace(lx, rx, N_ic).view(-1, 1).to(device)
        x_ic.requires_grad_(True)

        # “BC”：x = lx / rx 两条边上，t 从 ly 到 ry
        t_bc = torch.linspace(ly, ry, N_bc).view(-1, 1).to(device)
        t_bc.requires_grad_(True)

        return data, x_ic, t_bc


    def ic_loss(model, x_ic):
        # 拼出初始面的坐标 (x, t=ly)
        t_ic = ly * torch.ones_like(x_ic)
        X_ic = torch.cat([x_ic, t_ic], dim=1)   # shape (N_ic, 2)

        # 1) u(x, ly) ≈ u_exact(x, ly)
        u_pred = model(X_ic)
        u_true = u_exact_torch(X_ic)
        loss_ic = nn.MSELoss()(u_pred, u_true)

        # 2) u_t(x, ly) ≈ 0
        du = grad(outputs=u_pred,
                  inputs=X_ic,
                  grad_outputs=torch.ones_like(u_pred),
                  create_graph=True)[0]
        u_t = du[:, 1:2]   # 对第二个坐标 t 求导

        loss_ic += torch.mean(u_t**2)

        return loss_ic

    def bc_loss(model, t_bc):
        
        t = t_bc

        x_left = lx * torch.ones_like(t)
        x_right = rx * torch.ones_like(t)

        X_left = torch.cat([x_left, t], dim=1)    # (lx, t)
        X_right = torch.cat([x_right, t], dim=1)  # (rx, t)

        # 预测值
        u_left_pred = model(X_left)
        u_right_pred = model(X_right)

        # 真解边界值
        u_left_true = u_exact_torch(X_left)
        u_right_true = u_exact_torch(X_right)

        loss_bc = nn.MSELoss()(u_left_pred, u_left_true) \
                  + nn.MSELoss()(u_right_pred, u_right_true)

        return loss_bc

    # ========= 真解预测 + 画图 =========
    def compute_y_pred(model):
        X, T = np.meshgrid(x_exact, t_exact)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        X_tensor = torch.tensor(X_star, dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            u_pred = model(X_tensor).cpu().numpy().reshape(X.shape)
        return u_pred

    def plot_solution(model, epoch):
        x = x_exact
        t = t_exact
        u_real = u_sol

        X, T = np.meshgrid(x, t)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        X_tensor = torch.tensor(X_star, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            u_pred = model(X_tensor).cpu().numpy().reshape(u_real.shape)

        abs_err = np.abs(u_real - u_pred)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].pcolormesh(X, T, u_real, shading='auto', cmap='jet')
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title("True Solution")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("t")

        im1 = axes[1].pcolormesh(X, T, u_pred, shading='auto', cmap='jet')
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title(f"Predicted Solution (Epoch {epoch})")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("t")

        im2 = axes[2].pcolormesh(X, T, abs_err, shading='auto', cmap='jet')
        fig.colorbar(im2, ax=axes[2])
        axes[2].set_title("Absolute Error")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("t")

        plt.tight_layout()
        os.makedirs('solutions', exist_ok=True)
        plt.savefig(f'solutions/solution_epoch_{epoch}.png')
        plt.close(fig)

    def plot_residual_and_samples(model, data, epoch, resample_count):
        N = 100
        x = np.linspace(lx, rx, N)
        t = np.linspace(ly, ry, N)
        X, T = np.meshgrid(x, t)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

        model.eval()
        residuals = []
        batch_size = 1000

        for i in range(0, len(X_star), batch_size):
            batch = X_star[i:i + batch_size]
            X_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            # 绘图计算残差也需要梯度
            X_tensor.requires_grad_(True)

            with torch.set_grad_enabled(True):
                u_pred = model(X_tensor)
                res = pde_loss(u_pred, X_tensor)
                residual_values = (res ** 2).detach().cpu().numpy().squeeze()
            residuals.extend(list(residual_values))

        residuals = np.array(residuals)
        if residuals.size > len(X_star):
            residuals = residuals[:len(X_star)]
        elif residuals.size < len(X_star):
            residuals = np.concatenate(
                [residuals, np.full(len(X_star) - residuals.size,
                                    residuals.mean() if residuals.size > 0 else 0.0)]
            )

        residuals = residuals.reshape(X.shape)
        samples_actual = data.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        vmax = np.percentile(residuals, 95) if np.percentile(residuals, 95) > 0 else 1.0
        vmin = 0.0

        im1 = axes[0].pcolormesh(X, T, residuals, shading='auto', cmap='jet',
                                 vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=axes[0])
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('t')
        axes[0].set_title(f'PDE Residual Heatmap\nEpoch {epoch}, Resample {resample_count}')

        im2 = axes[1].pcolormesh(X, T, residuals, shading='auto', cmap='jet',
                                 vmin=vmin, vmax=vmax)
        plt.colorbar(im2, ax=axes[1])
        axes[1].scatter(samples_actual[:, 0], samples_actual[:, 1],
                        c='white', s=2, alpha=0.8, label='Samples')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('t')
        axes[1].set_title(f'Residual + Samples\nEpoch {epoch}, Resample {resample_count}')
        axes[1].legend()

        axes[2].scatter(samples_actual[:, 0], samples_actual[:, 1],
                        c='red', s=2, alpha=0.8)
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('t')
        axes[2].set_title(f'Sampling Points Only\nEpoch {epoch}, Resample {resample_count}')
        axes[2].set_xlim(lx, rx)
        axes[2].set_ylim(ly, ry)

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.set_xlim(lx, rx)
            ax.set_ylim(ly, ry)

        plt.tight_layout()
        os.makedirs('residual_samples', exist_ok=True)
        plt.savefig(f'residual_samples/residual_samples_epoch_{epoch}_resample_{resample_count}.png')
        plt.close(fig)

        print(f"Epoch {epoch}, Resample {resample_count}: "
              f"残差范围 [{residuals.min():.2e}, {residuals.max():.2e}], "
              f"采样点数量: {len(samples_actual)}")

    # ========= 初始化一次采样 =========
    data, x_ic, t_bc = get_samplers(Samples, N_f=num_r, N_ic=num_0, N_bc=num_b, T=20)
    plot_residual_and_samples(model, data, epoch=0, resample_count=0)

    u_true = u_sol

    # ========= 权重初始化 =========
    if method == 1:
        lambd_ic = torch.tensor(1.0, device=device, requires_grad=True)
        lambd_bc = torch.tensor(1.0, device=device, requires_grad=True)
    elif method == 2:
        lambd_ic = torch.tensor(1.0, device=device, requires_grad=False)
        lambd_bc = torch.tensor(1.0, device=device, requires_grad=False)
    elif method == 3:
        lambd_ic = torch.tensor(0.0, device=device, requires_grad=False)
        lambd_bc = torch.tensor(0.0, device=device, requires_grad=False)
    else:
        # method = 0 或其它情况：后面会按 equal weighting 逻辑处理
        lambd_ic = torch.tensor(1.0, device=device, requires_grad=False)
        lambd_bc = torch.tensor(1.0, device=device, requires_grad=False)

    params = [{'params': model.parameters(), 'lr': lr}]
    if guding_lr:
        optimizer = Adam(params, betas=(0.99, 0.999))
        scheduler = None
    else:
        optimizer = Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)

    loss_history = []
    loss_pde_history = []
    loss_bc_history = []
    loss_ic_history = []

    lambd_r = 1.0
    lambd_bc_history = []
    lambd_ic_history = []

    mean_loss_pde = 0.0
    mean_loss_bc = 0.0
    mean_loss_ic = 0.0

    resample_count = 0

    # ========= 训练循环 =========
    for epoch in range(Adam_n_epochs):
        model.train()
        optimizer.zero_grad()

        if not data.requires_grad:
            data.requires_grad_(True)

        # 定期重新采样 PDE 点
        if epoch % N_change_point == 0 and epoch != 0:
            data, x_ic, t_bc = get_samplers(Samples, N_f=num_r, N_ic=num_0, N_bc=num_b, T=5)
            plot_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)
            save_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)

        uhat = model(data)
        # PDE 残差平方的均值作为 PDE loss
        loss_pde = pde_loss(uhat, data).pow(2).mean()
        loss_ic = ic_loss(model, x_ic)
        loss_bc = bc_loss(model, t_bc)

        loss_pde_history.append(loss_pde.item())
        loss_bc_history.append(loss_bc.item())
        loss_ic_history.append(loss_ic.item())

        # 更新移动平均
        mean_loss_bc = (1 - 1 / (epoch + 1)) * mean_loss_bc + (1 / (epoch + 1)) * loss_bc.item()
        mean_loss_ic = (1 - 1 / (epoch + 1)) * mean_loss_ic + (1 / (epoch + 1)) * loss_ic.item()
        mean_loss_pde = (1 - 1 / (epoch + 1)) * mean_loss_pde + (1 / (epoch + 1)) * loss_pde.item()

        # 梯度统计
        maxr, meanr = loss_grad_max_mean(loss_pde, model, lambg=lambd_r)
        maxb, meanb = loss_grad_max_mean(loss_bc, model, lambg=lambd_bc)
        maxi, meani = loss_grad_max_mean(loss_ic, model, lambg=lambd_ic)

        # 动态调整权重（保持原脚本的策略）
        if epoch % N_change_lambda == 0 and epoch != 0:
            if method == 0:
                lambd_bc = torch.ones(1).to(device) * 6
                lambd_ic = torch.ones(1).to(device) * 6
            elif method == 1:
                hat_all = maxr / meanb + maxr / meani
                punish_bc = loss_bc.item() / mean_loss_bc
                punish_ic = loss_ic.item() / mean_loss_ic
                sum_punish = punish_bc + punish_ic + 1e-8
                lambd_bc = (1 - 1 / (epoch + 1)) * lambd_bc + (punish_bc / sum_punish) * hat_all * 1 / (epoch + 1)
                lambd_ic = (punish_ic / sum_punish) * hat_all * 1 / (epoch + 1) + (1 - 1 / (epoch + 1)) * lambd_ic
            elif method == 2:
                hat_all = maxr / meanb + maxr / meani
                punish_bc = loss_bc.item() / mean_loss_bc
                punish_ic = loss_ic.item() / mean_loss_ic
                sum_punish = punish_bc + punish_ic + 1e-8
                lambd_bc = 0.9 * lambd_bc + (punish_bc / sum_punish) * hat_all * 0.1
                lambd_ic = (punish_ic / sum_punish) * hat_all * 0.1 + 0.9 * lambd_ic
            # 其它方法（3, DB_PINN 等）如果你以后要加，可以按原 KG 文件扩展

        lambd_bc_history.append(lambd_bc.item())
        lambd_ic_history.append(lambd_ic.item())

        # 总损失
        loss = loss_pde + lambd_bc.item() * loss_bc + lambd_ic.item() * loss_ic
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        if guding_lr:
            optimizer.step()
        else:
            optimizer.step()
            scheduler.step()

        if epoch % 10 == 0:
            print('Epoch: %d, Loss: %.5e, Loss_pde: %.5e, Loss_bc: %.5e, Loss_ic: %.5e, lr: %.3e' %
                  (epoch, loss.item(), loss_pde.item(), loss_bc.item(), loss_ic.item(),
                   optimizer.param_groups[0]['lr']))

            u_pred = compute_y_pred(model)
            L2_mean, L2_max, L2_min = compute_L2_error(u_true, u_pred)
            print(f"    L2 error - Mean: {L2_mean:.3e}, Max: {L2_max:.3e}, Min: {L2_min:.3e}")
            print('    lambd_bc: %.3e, lambd_ic: %.3e' % (lambd_bc.item(), lambd_ic.item()))

        if epoch % 60 == 0:
            plot_residual_and_samples(model, data, epoch, resample_count)
            plot_solution(model, epoch)

    # 训练结束后存下 loss 曲线
    plot_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)
    save_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)

    return model


if __name__ == "__main__":
    # 想和第一个文件一样的定义域，就用 [0,1]×[0,1]
    default_bounds = [[0.0, 1.0], [0.0, 1.0]]
    trained_model = train(bounds=default_bounds, method=1)