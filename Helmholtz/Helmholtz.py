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
import torch
import numpy as np
import random
from collections import deque
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

def plot_L2_history(l2_history):
    plt.figure(figsize=(10, 6))
    plt.semilogy(l2_history, label='L2 Error')
    plt.xlabel('Epoch (every 10 epochs)')
    plt.ylabel('L2 Error')
    plt.title('L2 Error History')
    plt.legend()
    plt.grid()
    plt.savefig('L2_error_history.png')
    plt.close()
def save_L2_history(l2_history, filename='L2_error_history.npz'):
    np.savez(filename, l2_history=np.array(l2_history))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.set_printoptions(threshold=np.inf)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-3
Adam_n_epochs = 20001
num_0 = 200      # 边界/“IC” 点数（上下边）
num_b = 200      # 边界点数（左右边）
num_r = 10000     # PDE 内部点
dim = 2
num_hidden = 50
num_layers = 7
layer_sizes = [dim] + [num_hidden] * num_layers + [1]

N_change_point = 50     # 每隔多少 epoch 重新采样
N_change_lambda = 10    # 每隔多少 epoch 更新 loss 权重


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


set_seed(2006)


def compute_L2_error(y_true, y_pred):

    num = np.sqrt(np.mean((y_true - y_pred)**2))
    den = np.sqrt(np.mean(y_true**2)) + 1e-12
    return num / den,0,0


# experiment setup
def train(bounds, method=1, guding_lr=False):
 
    lx, rx = bounds[0]
    ly, ry = bounds[1]

    print(f"Using domain: x in [{lx}, {rx}], y in [{ly}, {ry}]")
    print("Network layers:", layer_sizes)

    # -------- PDE & 真解参数 --------
    a_1 = 1
    a_2 = 4
    k = 1
    lam = k ** 2

    # 真解和一些辅助函数（依赖 bounds）
    def generate_u(x):
        return np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

    def u_xx(x):
        return - (a_1 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

    def u_yy(x):
        return - (a_2 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

    def forcing(x):
        return u_xx(x) + u_yy(x) + lam * generate_u(x)

    # 生成真解网格
    x_exact = np.linspace(lx, rx, 1001)[:, None]
    y_exact = np.linspace(ly, ry, 1001)[:, None]
    xx, yy = np.meshgrid(x_exact, y_exact)
    X_grid = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
    u_sol = generate_u(X_grid).reshape(xx.shape)

    # -------- PINN 模型 --------
    model = PINN(sizes=layer_sizes, activation=torch.nn.Tanh()).to(device)

    # -------- PDE 残差（Helmholtz）--------
    def pde_loss(uhat, data):
        x = data[:, 0:1]
        y = data[:, 1:2]

        du = grad(outputs=uhat, inputs=data,
                  grad_outputs=torch.ones_like(uhat),
                  create_graph=True)[0]
        dudx = du[:, 0:1]
        dudy = du[:, 1:2]

        dudxx = grad(outputs=dudx, inputs=data,
                     grad_outputs=torch.ones_like(dudx),
                     create_graph=True)[0][:, 0:1]
        dudyy = grad(outputs=dudy, inputs=data,
                     grad_outputs=torch.ones_like(dudy),
                     create_graph=True)[0][:, 1:2]

        source = - (a_1 * math.pi) ** 2 * torch.sin(a_1 * math.pi * x) * torch.sin(a_2 * math.pi * y) \
                 - (a_2 * math.pi) ** 2 * torch.sin(a_1 * math.pi * x) * torch.sin(a_2 * math.pi * y) \
                 + lam * torch.sin(a_1 * math.pi * x) * torch.sin(a_2 * math.pi * y)

        residual = dudxx + dudyy + lam * uhat - source
        return residual


    def loss_fn_for_sampler(xy):
     
        with torch.set_grad_enabled(True):
            if not xy.requires_grad:
                xy.requires_grad_(True)
            
            uhat = model(xy)
            res = pde_loss(uhat, xy)
            
            # 确保返回的是一维 Tensor
            return (res ** 2).flatten()

    # -------- IDEALSampler 实例（用 bounds）--------
    Samples = IDEALSampler(
        loss_fn=loss_fn_for_sampler,
        bounds=bounds,
        device=device
    )

    # -------- 采样器包装 --------
    def get_samplers(Samples, N_f=10000, N_ic=100, N_bc=100, T=20):
        """
        Samples: IDEALSampler 实例
        在整个区域 [lx,rx]×[ly,ry] 上采样：
        - N_f 个 PDE 内部点（通过 IDEALSampler）
        - N_ic: 用于 y=ly, y=ry 两条边
        - N_bc: 用于 x=lx, x=rx 两条边
        """
        # 内部点初始分布：全域均匀
        init_f = np.random.uniform([lx, ly], [rx, ry], (N_f, 2))

        data_np = Samples.sample(n_samples=N_f, init_points=init_f, generations=T)
        data = torch.tensor(data_np, dtype=torch.float32, device=device)
        data.requires_grad_(True)

        # 上下边界 y = ly, y = ry
        x_ic = torch.linspace(lx, rx, N_ic).view(-1, 1).to(device)
        x_ic.requires_grad_(True)

        # 左右边界 x = lx, x = rx （这里 t_bc 本质上是 y）
        t_bc = torch.linspace(ly, ry, N_bc).view(-1, 1).to(device)
        t_bc.requires_grad_(True)

        return data, x_ic, t_bc

    # -------- 边界损失（Dirichlet: u=0）--------
    def ic_loss(model, x_ic):
        """
        y = ly 和 y = ry 边界
        """
        y_bottom = ly * torch.ones_like(x_ic)
        y_top = ry * torch.ones_like(x_ic)

        X_bottom = torch.cat([x_ic, y_bottom], dim=1)
        X_top = torch.cat([x_ic, y_top], dim=1)

        u_bottom = model(X_bottom)
        u_top = model(X_top)

        zeros_bottom = torch.zeros_like(u_bottom)
        zeros_top = torch.zeros_like(u_top)

        loss_ic = nn.MSELoss()(u_bottom, zeros_bottom) + nn.MSELoss()(u_top, zeros_top)
        return loss_ic

    def bc_loss(model, t_bc):
        """
        x = lx 和 x = rx 边界，t_bc 在这里就是 y
        """
        y_bc = t_bc

        x_left = lx * torch.ones_like(y_bc)
        x_right = rx * torch.ones_like(y_bc)

        X_left = torch.cat([x_left, y_bc], dim=1)
        X_right = torch.cat([x_right, y_bc], dim=1)

        u_left = model(X_left)
        u_right = model(X_right)

        zeros_left = torch.zeros_like(u_left)
        zeros_right = torch.zeros_like(u_right)

        loss_bc = nn.MSELoss()(u_left, zeros_left) + nn.MSELoss()(u_right, zeros_right)
        return loss_bc

    # -------- 真解预测 + 可视化 --------
    def compute_y_pred(model):
        X, Y = np.meshgrid(x_exact, y_exact)
        X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
        X_tensor = torch.tensor(X_star, dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            u_pred = model(X_tensor).cpu().numpy().reshape(X.shape)
        return u_pred

    def plot_solution(model, epoch):
        x = x_exact
        y = y_exact
        u_real = u_sol

        X, Y = np.meshgrid(x, y)
        X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
        X_tensor = torch.tensor(X_star, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            u_pred = model(X_tensor).cpu().numpy().reshape(u_real.shape)

        abs_err = np.abs(u_real - u_pred)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].pcolormesh(X, Y, u_real, shading='auto', cmap='jet')
        fig.colorbar(im0, ax=axes[0])
        axes[0].set_title("True Solution")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        im1 = axes[1].pcolormesh(X, Y, u_pred, shading='auto', cmap='jet')
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_title(f"Predicted Solution (Epoch {epoch})")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        im2 = axes[2].pcolormesh(X, Y, abs_err, shading='auto', cmap='jet')
        fig.colorbar(im2, ax=axes[2])
        axes[2].set_title("Absolute Error")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")

        plt.tight_layout()
        os.makedirs('solutions', exist_ok=True)
        plt.savefig(f'solutions/solution_epoch_{epoch}.png')
        plt.close(fig)

    def plot_residual_and_samples(model, data, epoch, resample_count):
        N = 100
        x = np.linspace(lx, rx, N)
        y = np.linspace(ly, ry, N)
        X, Y = np.meshgrid(x, y)
        X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

        model.eval()
        residuals = []
        batch_size = 1000

        for i in range(0, len(X_star), batch_size):
            batch = X_star[i:i + batch_size]
            X_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            X_tensor.requires_grad_(True)

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

        im1 = axes[0].pcolormesh(X, Y, residuals, shading='auto', cmap='jet',
                                 vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=axes[0])
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title(f'PDE Residual Heatmap\nEpoch {epoch}, Resample {resample_count}')

        im2 = axes[1].pcolormesh(X, Y, residuals, shading='auto', cmap='jet',
                                 vmin=vmin, vmax=vmax)
        plt.colorbar(im2, ax=axes[1])
        axes[1].scatter(samples_actual[:, 0], samples_actual[:, 1],
                        c='white', s=2, alpha=0.8, label='Samples')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title(f'Residual + Samples\nEpoch {epoch}, Resample {resample_count}')
        axes[1].legend()

        axes[2].scatter(samples_actual[:, 0], samples_actual[:, 1],
                        c='red', s=2, alpha=0.8)
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
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

    # 先初始化一次采样
    data, x_ic, t_bc = get_samplers(Samples, N_f=num_r, N_ic=num_0, N_bc=num_b, T=20)
    plot_residual_and_samples(model, data, epoch=0, resample_count=0)

    u_true = u_sol

    # -------- 权重初始化 --------
    if method == 0:
        lambd_ic = torch.tensor(1.0, device=device, requires_grad=True)
        lambd_bc = torch.tensor(1.0, device=device, requires_grad=True)
    elif method == 1:
        lambd_ic = torch.tensor(1.0, device=device, requires_grad=False)
        lambd_bc = torch.tensor(1.0, device=device, requires_grad=False)
    elif method == 2:
        lambd_ic = torch.tensor(0.0, device=device, requires_grad=False)
        lambd_bc = torch.tensor(0.0, device=device, requires_grad=False)
    else:
        raise ValueError("未知方法 method")

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
    L2_mean_history = []

    lambd_r = 1.0

    lambd_bc_history = []
    lambd_ic_history = []

    mean_loss_pde = 0.0
    mean_loss_bc = 0.0
    mean_loss_ic = 0.0

    resample_count = 0

    # -------- 训练循环 --------
    for epoch in range(Adam_n_epochs):
        model.train()
        optimizer.zero_grad()
        
        if not data.requires_grad:
            data.requires_grad_(True)
        
        if epoch % N_change_point == 0 and epoch != 0:
            data, x_ic, t_bc = get_samplers(Samples, N_f=num_r, N_ic=num_0, N_bc=num_b,T= 5)
            plot_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)
            save_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)
            plot_L2_history(L2_mean_history)
            save_L2_history(L2_mean_history)

        uhat = model(data)
        loss_pde = pde_loss(uhat, data).pow(2).mean()
        loss_ic = ic_loss(model, x_ic)
        loss_bc = bc_loss(model, t_bc)
        
        loss_pde_history.append(loss_pde.item())
        loss_bc_history.append(loss_bc.item())
        loss_ic_history.append(loss_ic.item())

        # #更新平均损失
        # mean_loss_bc = (1 - 1/(epoch+1)) * mean_loss_bc + (1/(epoch+1)) * loss_bc.item()
        # mean_loss_ic = (1 - 1/(epoch+1)) * mean_loss_ic + (1/(epoch+1)) * loss_ic.item()
        # mean_loss_pde = (1 - 1/(epoch+1)) * mean_loss_pde + (1/(epoch+1)) * loss_pde.item()

        # # stdr, kurtr = loss_grad_stats(loss_pde, model)
        # # stdb, kurtb = loss_grad_stats(loss_bc, model)
        # # stdi, kurti = loss_grad_stats(loss_ic, model)

        # maxr, meanr = loss_grad_max_mean(loss_pde, model, lambg=lambd_r)
        # maxb, meanb = loss_grad_max_mean(loss_bc, model, lambg=lambd_bc)
        # maxi, meani = loss_grad_max_mean(loss_ic, model, lambg=lambd_ic)

        
        if epoch % N_change_lambda == 0 and epoch != 0:

            if method == 0:
                lambd_bc = torch.ones(1).to(device)*6
                lambd_ic = torch.ones(1).to(device)*6
            elif method == 1:
                hat_all = maxr/meanb + maxr/meani 
                punish_bc = loss_bc.item()/mean_loss_bc
                punish_ic = loss_ic.item()/mean_loss_ic

                sum_punish = punish_bc + punish_ic + 1e-8
                lambd_bc = (1-1/(epoch+1))*lambd_bc + (punish_bc / sum_punish) * hat_all * 1/(epoch+1)
                lambd_ic = (punish_ic / sum_punish) * hat_all * 1/(epoch+1) + (1-1/(epoch+1))*lambd_ic
            elif method == 2:
                hat_all = maxr/meanb + maxr/meani 
                punish_bc = loss_bc.item()/mean_loss_bc
                punish_ic = loss_ic.item()/mean_loss_ic

                sum_punish = punish_bc + punish_ic + 1e-8
                lambd_bc = 0.9*lambd_bc + (punish_bc / sum_punish) * hat_all * 0.1
                lambd_ic = (punish_ic / sum_punish) * hat_all * 0.1 + 0.9*lambd_ic
                
        lambd_bc_history.append(lambd_bc.item())
        lambd_ic_history.append(lambd_ic.item())
        
        # 计算总损失
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
            L2_mean_history.append(L2_mean)
            print(f"    L2 error - Mean: {L2_mean:.3e}, Max: {L2_max:.3e}, Min: {L2_min:.3e}")
            print('    lambd_bc: %.3e, lambd_ic: %.3e' % (lambd_bc.item(), lambd_ic.item()))
        
        if epoch % 60 == 0:
            plot_residual_and_samples(model, data, epoch, resample_count)
            plot_solution(model, epoch)
    plot_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)
    save_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)

    

    return model


if __name__ == "__main__":
    default_bounds = [[-1.0, 1.0], [-1.0, 1.0]]
    trained_model = train(bounds=default_bounds, method=0)