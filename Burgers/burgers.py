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
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pinn import PINN
import torch
# [修改点 1] 导入 IDEALSampler
from ideal import IDEALSampler
import scipy.io as sio

b_data = sio.loadmat("burgers_shock.mat")
print(b_data.keys())
x_exact = b_data["x"].flatten()        # (Nx,)
t_exact = b_data["t"].flatten()       # (Nt,)
u_exact_mat = b_data["usol"]                 # (Nx, Nt)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(204)

def plot_solution(model, epoch):
   
    x = x_exact
    t = t_exact
    u_real = u_exact_mat.T      # shape = (Nt, Nx)

    # 生成预测点
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    X_tensor = torch.tensor(X_star, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        u_pred = model(X_tensor).cpu().numpy().reshape(u_real.shape)

    # 误差
    abs_err = np.abs(u_real - u_pred)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 图 1：真实解 ---
    im0 = axes[0].pcolormesh(X, T, u_real, shading='auto', cmap='jet')
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("True Solution (burgers.mat)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")

    # --- 图 2：预测解 ---
    im1 = axes[1].pcolormesh(X, T, u_pred, shading='auto', cmap='jet')
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title(f"Predicted Solution (epoch {epoch})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")

    # --- 图 3：绝对误差 ---
    im2 = axes[2].pcolormesh(X, T, abs_err, shading='auto', cmap='jet')
    fig.colorbar(im2, ax=axes[2])
    axes[2].set_title("|Prediction - True|")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")

    plt.tight_layout()
    plt.savefig(f"yhat_results/solution_compare_epoch_{epoch}.png")
    plt.close()

    print(f"[Info] solution_compare_epoch_{epoch}.png saved.")

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

def init_cuda_context():
    """修复CUDA上下文初始化"""
    if torch.cuda.is_available():
        # 清空缓存
        torch.cuda.empty_cache()
        
        # 设置设备
        device = torch.device('cuda')
        
        # 使用上下文管理器确保CUDA操作在正确的设备上
        with torch.cuda.device(device):
            # 执行实际的CUDA操作来建立上下文
            for _ in range(3):
                x = torch.randn(100, 100, device=device)
                y = torch.randn(100, 100, device=device)
                z = torch.mm(x, y)
                torch.cuda.synchronize()
        
        print(f"CUDA设备已稳定初始化: {torch.cuda.get_device_name(device)}")
        return device
    else:
        print("使用CPU")
        return torch.device('cpu')

device = init_cuda_context()

def plot_residual_and_samples(model, data, epoch, resample_count):
    # 创建网格
    N = 100
    x = np.linspace(-1, 1, N)
    t = np.linspace(0, 1, N)
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    
    model.eval()
    residuals = []
    
    batch_size = 1000
    for i in range(0, len(X_star), batch_size):
        batch = X_star[i:i+batch_size]
        X_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
        
        try:
            # 关键修复：正确设置梯度计算
            X_tensor.requires_grad_(True)
            
            with torch.enable_grad():
                # 前向传播
                u_pred = model(X_tensor)
                
                gradients = grad(outputs=u_pred, inputs=X_tensor,
                             grad_outputs=torch.ones_like(u_pred),
                             create_graph=True, retain_graph=True,
                             allow_unused=True)[0]  # 关键：添加allow_unused=True
                
                if gradients is not None:
                    # 从完整梯度中提取偏导数
                    dudx = gradients[:, 0:1]
                    dudt = gradients[:, 1:2]
                    
                    
                    dudx_x = grad(outputs=dudx, inputs=X_tensor,
                                grad_outputs=torch.ones_like(dudx),
                                create_graph=True, retain_graph=True,
                                allow_unused=True)[0]
                    
                    if dudx_x is not None:
                        dudxx = dudx_x[:, 0:1]
                    else:
                        # 如果二阶导计算失败，使用零张量
                        dudxx = torch.zeros_like(dudx)
                    
                    # 计算PDE残差（Burgers方程）
                    residual = dudt + 1.0 * u_pred * dudx - D_nu * dudxx
                    residual_values = (residual ** 2).squeeze()
                    
                else:
                    residual_values = torch.ones(X_tensor.shape[0]) * 0.01
                
        except Exception as e:
            print(f"梯度计算错误: {e}")
            with torch.no_grad():
                u_simple = model(X_tensor)
                residual_values = torch.abs(u_simple).squeeze() * 0.1  
        
       
        residuals.extend(residual_values.detach().cpu().numpy())
    
   
    if len(residuals) > len(X_star):
        residuals = residuals[:len(X_star)]
    elif len(residuals) < len(X_star):
        residuals.extend([0.01] * (len(X_star) - len(residuals)))
    
    residuals = np.array(residuals).reshape(X.shape)
    
    samples_actual = data.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: 残差热力图
    vmax = np.percentile(residuals, 95) if np.percentile(residuals, 95) > 0 else 1.0
    vmin = 0
    
    im1 = axes[0].pcolormesh(X, T, residuals, shading='auto', cmap='jet', 
                           vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=axes[0])
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title(f'PDE Residual Heatmap\nEpoch {epoch}, Resample {resample_count}')
    
    # 图2: 残差热力图 + 采样点
    im2 = axes[1].pcolormesh(X, T, residuals, shading='auto', cmap='jet', 
                           vmin=vmin, vmax=vmax)
    plt.colorbar(im2, ax=axes[1])
    axes[1].scatter(samples_actual[:, 0], samples_actual[:, 1], 
                   c='white', s=2, alpha=0.8, label='Samples')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title(f'Residual + Samples\nEpoch {epoch}, Resample {resample_count}')
    axes[1].legend()
    
    # 图3: 只有采样点
    axes[2].scatter(samples_actual[:, 0], samples_actual[:, 1], 
                   c='red', s=2, alpha=0.8)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    axes[2].set_title(f'Sampling Points Only\nEpoch {epoch}, Resample {resample_count}')
    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(0, 1)
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'residual_samples/residual_samples_epoch_{epoch}_resample_{resample_count}.png')
    print(f"Epoch {epoch}, Resample {resample_count}: "
          f"残差范围 [{residuals.min():.2e}, {residuals.max():.2e}], "
          f"采样点数量: {len(samples_actual)}")
    
lr = 1e-3   
Adam_n_epochs   = 8001
layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
num_0 = 100
num_b = 100   
num_r = 10000
num_u = 300
D_nu = 0.01/np.pi

def pde_loss(uhat, data, lam1, lam2):  # data: (x, t)
    x = data[:,0:1]
    t = data[:,1:2]

    if not data.requires_grad:
        data.requires_grad_(True)
    
    du = grad(outputs=uhat, inputs=data, 
              grad_outputs=torch.ones_like(uhat), create_graph=True, 
              allow_unused=True)[0]
    
    if du is None:
        return torch.tensor(0.0, device=data.device, requires_grad=True)
    
    dudx = du[:,0:1]
    dudt = du[:,1:2]
    
    dudxx = grad(outputs=dudx, inputs=data, 
                 grad_outputs=torch.ones_like(dudx), create_graph=True, 
                 allow_unused=True)[0]
    
    if dudxx is None:
        return torch.tensor(0.0, device=data.device, requires_grad=True)
    
    dudxx = dudxx[:,0:1]
                                                                                                                             
    residual = dudt + lam1*uhat*dudx - lam2*dudxx
    loss_pde = torch.mean(residual**2)

    return loss_pde

def loss_fn(xy, model=None):
    if isinstance(xy, np.ndarray):
        xy_tensor = torch.from_numpy(xy).float().to(device)
    else:
        xy_tensor = xy

    with torch.set_grad_enabled(True):
        
        if not xy_tensor.requires_grad:
            xy_tensor.requires_grad_(True)

        u = model(xy_tensor)

        du = grad(
            outputs=u,
            inputs=xy_tensor,
            grad_outputs=torch.ones_like(u),
            create_graph=True, 
            retain_graph=True
        )[0]

        dudx = du[:, 0:1]
        dudt = du[:, 1:2]

        d2u_dx2 = grad(
            outputs=dudx,
            inputs=xy_tensor,
            grad_outputs=torch.ones_like(dudx),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]

      
        residual = dudt + u * dudx - D_nu * d2u_dx2
        
        
        loss_squared = (residual ** 2).squeeze()
        
    return loss_squared

def update_model(model,RL_model, episodes=200, T=20):
    
    loss = lambda xy: loss_fn(xy, model=model)
   
    RL_model.loss_fn = loss
      
def get_samplers(RL_model, N_f=10000, N_ic=100, N_bc=100,T = 20):

   
    init_f = np.random.uniform([-1,0],[1,1], (N_f,2))
   
    data = RL_model.sample(n_samples=N_f, init_points=init_f, generations=T)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    data.requires_grad_(True)


    x_ic = torch.linspace(-1, 1, N_ic).view(-1,1).to(device)
    x_ic.requires_grad_(True)

   
    t_bc = torch.linspace(0,1, N_bc).view(-1,1).to(device)
    t_bc.requires_grad_(True)

    return data, x_ic, t_bc

def ic_loss(model, x_ic): 
    t_ic = torch.zeros_like(x_ic)
    X_ic = torch.cat([x_ic, t_ic], dim=1)
    u0_pred = model(X_ic)
    loss_ic = nn.MSELoss()(u0_pred, -torch.sin(np.pi * x_ic))
    return loss_ic

def bc_loss(model, t_bc):
    x_left = -torch.ones_like(t_bc)
    x_right = torch.ones_like(t_bc)
    X_left = torch.cat([x_left, t_bc], dim=1)
    X_right = torch.cat([x_right, t_bc], dim=1)
    u_left = model(X_left)
    u_right = model(X_right)
    loss_bc = nn.MSELoss()(u_left, torch.zeros_like(u_left)) + \
               nn.MSELoss()(u_right, torch.zeros_like(u_right))
    return loss_bc


N_change_point = 20
N_update_RL = 10000
N_change_lambda = 10

model = PINN(sizes=layer_sizes, activation=torch.nn.Tanh()).to(device)

RL_samples = IDEALSampler(
    loss_fn=lambda xy: loss_fn(xy, model),
    bounds=[[-1.0, 1.0], [0.0, 1.0]],
    device=device
)

update_model(model, RL_samples, episodes=400, T=25)
data, x_ic, t_bc = get_samplers(RL_samples, N_f=num_r, N_ic=num_0, N_bc=num_b,T= 20)
plot_residual_and_samples(model, data, epoch=0, resample_count=0)

def train(data = data,x_ic=x_ic,t_bc=t_bc,guding_lr=False, method=0):
    lambd_bc = torch.ones(1).to(device)
    lambd_ic = torch.ones(1).to(device)
    lambd_r = torch.ones(1).to(device)
    lambd_bc_history = []
    lambd_ic_history = []
    loss_history = []
    loss_pde_history = []
    loss_bc_history = []
    loss_ic_history = []

    mean_loss_pde = 0
    mean_loss_bc = 0
    mean_loss_ic = 0

    params = [{'params': model.parameters(), 'lr': lr}] 
    if guding_lr:
        optimizer = Adam(params, betas=(0.99, 0.999)) 
    else:
        optimizer = Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9) 
    
    resample_count = 0 
    
    for epoch in range(Adam_n_epochs):
        model.train()
        optimizer.zero_grad()
        
        if not data.requires_grad:
            data.requires_grad_(True)
        
        if epoch % N_update_RL == 0 and epoch != 0:
            update_model(model, RL_samples, episodes=50, T=5)
        
        if epoch % N_change_point == 0 and epoch != 0:
            data, x_ic, t_bc = get_samplers(RL_samples, N_f=num_r, N_ic=num_0, N_bc=num_b,T= 5)
            plot_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)
            save_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)

        uhat = model(data)
        loss_pde = pde_loss(uhat, data, 1.0, D_nu)
        loss_ic = ic_loss(model, x_ic)
        loss_bc = bc_loss(model, t_bc)
        
        loss_pde_history.append(loss_pde.item())
        loss_bc_history.append(loss_bc.item())
        loss_ic_history.append(loss_ic.item())

        #更新平均损失
        mean_loss_bc = (1 - 1/(epoch+1)) * mean_loss_bc + (1/(epoch+1)) * loss_bc.item()
        mean_loss_ic = (1 - 1/(epoch+1)) * mean_loss_ic + (1/(epoch+1)) * loss_ic.item()
        mean_loss_pde = (1 - 1/(epoch+1)) * mean_loss_pde + (1/(epoch+1)) * loss_pde.item()

        stdr, kurtr = loss_grad_stats(loss_pde, model)
        stdb, kurtb = loss_grad_stats(loss_bc, model)
        stdi, kurti = loss_grad_stats(loss_ic, model)

        maxr, meanr = loss_grad_max_mean(loss_pde, model, lambg=lambd_r)
        maxb, meanb = loss_grad_max_mean(loss_bc, model, lambg=lambd_bc)
        maxi, meani = loss_grad_max_mean(loss_ic, model, lambg=lambd_ic)

        
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
            print('    lambd_bc: %.3e, lambd_ic: %.3e' % (lambd_bc.item(), lambd_ic.item()))
        
        if epoch % 60 == 0:
            plot_residual_and_samples(model, data, epoch, resample_count)
            plot_solution(model, epoch)
    plot_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)
    save_loss_history(loss_history, loss_pde_history, loss_bc_history, loss_ic_history)

    return

train(method=1,data=data,x_ic=x_ic,t_bc=t_bc)