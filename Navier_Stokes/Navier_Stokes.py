import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import scipy.io
import pandas as pd

from ideal import IDEALSampler
from pinn import PINN
from grad_stats import *

# ===========================
#        配置与设置
# ===========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.set_printoptions(threshold=np.inf)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练超参数
lr = 1e-3
Adam_n_epochs = 50001
num_r = 20000    # PDE 内部点 (IDEAL 采样)
num_b = 1000      # 每个边界的点数 (4条边)

# 网络结构: 输入 (x,y) -> 输出 (psi, p)
# psi: 流函数, p: 压力
dim_in = 2
dim_out = 2
num_hidden = 100
num_layers = 8
layer_sizes = [dim_in] + [num_hidden] * num_layers + [dim_out]

# 物理参数
Re = 100.0       # 雷诺数
lx, rx = 0.0, 1.0
ly, ry = 0.0, 1.0
bounds = [[lx, rx], [ly, ry]]

# 采样与更新频率
N_change_point = 50    # 每隔多少 epoch 重新采样 PDE 点
N_change_lambda = 10    # 每隔多少 epoch 更新 loss 权重

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(1234)

# ===========================
#      数据加载 (CSV)
# ===========================
# 请修改这里的路径为你实际 CSV 文件的位置
# 假设数据在 ../data/Lid-driven-Cavity/ 下，或者你填写绝对路径
DATA_PATH_U = "Lid-driven-Cavity\\reference_u.csv"
DATA_PATH_V = "Lid-driven-Cavity\\reference_v.csv"

# 为了防止报错，这里做一个简单的路径检查，如果没有文件则生成假数据用于测试代码
try:
    u_ref = np.genfromtxt(DATA_PATH_U, delimiter=',')
    v_ref = np.genfromtxt(DATA_PATH_V, delimiter=',')
    print("Reference data loaded successfully.")
except:
    print(f"Warning: Data files not found at {DATA_PATH_U}. Using dummy data for shape.")
    u_ref = np.zeros((100, 100))
    v_ref = np.zeros((100, 100))

velocity_ref = np.sqrt(u_ref**2 + v_ref**2)

# 构建网格 (用于计算 L2 error 和画图)
nx, ny = u_ref.shape
x = np.linspace(lx, rx, nx)
y = np.linspace(ly, ry, ny)
X, Y = np.meshgrid(x, y)
X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

# 展平的参考解
u_sol = u_ref.T.flatten()[:, None] # 注意转置，匹配 meshgrid 顺序
v_sol = v_ref.T.flatten()[:, None]
velocity_sol = velocity_ref.T.flatten()[:, None]

# ===========================
#      物理方程定义
# ===========================

# 1. 从网络输出计算物理量 u, v, p, psi
def NS_uv(uhat, data):
    """
    data: (x,y)
    uhat: (psi, p)
    u = psi_y, v = -psi_x
    """
    psi = uhat[:, 0:1]
    p   = uhat[:, 1:2]
    
    # 自动微分求导
    psi_xy = grad(outputs=psi, inputs=data, 
                  grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    
    u = psi_xy[:, 1:2]   # psi_y
    v = -psi_xy[:, 0:1]  # -psi_x
    return u, v, p

# 2. 计算 PDE 残差
def NS_res(uhat, data):
    """
    Momentum Equations:
    u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy) = 0
    u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy) = 0
    """
    x = data[:, 0:1]
    y = data[:, 1:2]
    
    u, v, p = NS_uv(uhat, data)
    
    # 一阶导
    u_xy = grad(u, data, torch.ones_like(u), create_graph=True)[0]
    u_x = u_xy[:, 0:1]
    u_y = u_xy[:, 1:2]
    
    v_xy = grad(v, data, torch.ones_like(v), create_graph=True)[0]
    v_x = v_xy[:, 0:1]
    v_y = v_xy[:, 1:2]
    
    p_xy = grad(p, data, torch.ones_like(p), create_graph=True)[0]
    p_x = p_xy[:, 0:1]
    p_y = p_xy[:, 1:2]
    
    # 二阶导
    u_xx = grad(u_x, data, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = grad(u_y, data, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    
    v_xx = grad(v_x, data, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = grad(v_y, data, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
    
    # 残差
    f_u = u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re
    f_v = u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re
    
    return f_u, f_v

# 3. 采样器专用 Loss
def loss_fn_for_sampler(xy):
    # 强制开启梯度
    with torch.set_grad_enabled(True):
        if not xy.requires_grad:
            xy.requires_grad_(True)
        
        uhat = model(xy)
        f_u, f_v = NS_res(uhat, xy)
        
        # 返回总残差平方 (f_u^2 + f_v^2)
        return (f_u**2 + f_v**2).flatten()

# ===========================
#      模型与采样器初始化
# ===========================
model = PINN(sizes=layer_sizes, activation=torch.nn.Tanh()).to(device)

Samples = IDEALSampler(
    loss_fn=loss_fn_for_sampler,
    bounds=bounds,
    device=device,
    alpha=2.0,       # 稳态流体推荐 2.0
    noise_std=0.05,  # 增加一点噪声防止死锁
    step_size=0.01
)

# ===========================
#      辅助函数
# ===========================

def get_samplers(Samples, N_f=10000, N_b=100, T=20):
    """
    生成：
    1. PDE 内部点 (通过 IDEAL 采样)
    2. 边界点 (Wall: u=0, v=0; Lid: u=1, v=0)
    """
    # 1. 内部点 (IDEAL)
    init_f = np.random.uniform([lx, ly], [rx, ry], (N_f, 2))
    data_np = Samples.sample(n_samples=N_f, init_points=init_f, generations=T)
    X_f = torch.tensor(data_np, dtype=torch.float32, device=device)
    X_f.requires_grad_(True)
    
    # 2. 边界点 (LHS 随机采样)
    # Wall: Left, Right, Bottom
    # Lid: Top
    
    # 生成各边坐标
    # Left: x=0, y=[0,1]
    y_b = np.random.rand(N_b, 1)
    X_left = np.hstack((np.zeros_like(y_b), y_b))
    
    # Right: x=1, y=[0,1]
    X_right = np.hstack((np.ones_like(y_b), y_b))
    
    # Bottom: x=[0,1], y=0
    x_b = np.random.rand(N_b, 1)
    X_bottom = np.hstack((x_b, np.zeros_like(x_b)))
    
    # Top (Lid): x=[0,1], y=1
    X_top = np.hstack((x_b, np.ones_like(x_b)))
    
    # 合并 Walls
    X_walls = np.vstack((X_left, X_right, X_bottom))
    
    # 转 Tensor
    data_walls = torch.tensor(X_walls, dtype=torch.float32, device=device)
    data_walls.requires_grad_(True)
    
    data_lid = torch.tensor(X_top, dtype=torch.float32, device=device)
    data_lid.requires_grad_(True)
    
    return X_f, data_walls, data_lid

# 请将此函数插入到您代码中合适的位置
def compute_L2_error(model):
    """
    计算标准的相对 L2 误差：||y_true - y_pred||_2 / ||y_true||_2
    （基于 Frobenius 范数，解决过零点处的数值爆炸问题）
    """
    # 1. 准备数据和预测
    X_tensor = torch.tensor(X_star, dtype=torch.float32, device=device)
    X_tensor.requires_grad_(True) # 需要梯度是因为 NS_uv 需要求导

    model.eval()
    uhat = model(X_tensor)
    # NS_uv 内部求导，返回 u, v, p 的预测值
    u_pred_t, v_pred_t, _ = NS_uv(uhat, X_tensor)
    
    # 2. 展平并转换为 Numpy (方便计算范数)
    # 关键：全部展平为一维向量，防止维度广播错误
    u_pred_flat = u_pred_t.detach().cpu().numpy().reshape(-1)
    v_pred_flat = v_pred_t.detach().cpu().numpy().reshape(-1)
    
    # 计算速度模长 (Scalar Field)
    vel_pred_flat = np.sqrt(u_pred_flat**2 + v_pred_flat**2)
    
    # 展平真值 (u_sol, v_sol, velocity_sol 是全局变量)
    u_true_flat = u_sol.reshape(-1)
    v_true_flat = v_sol.reshape(-1)
    vel_true_flat = velocity_sol.reshape(-1)
    
    eps = 1e-12

    def safe_rel_l2(y_true, y_pred):
        """核心计算: ||Error||_2 / (||True||_2 + eps)"""
        error_norm = np.linalg.norm(y_true - y_pred)
        true_norm = np.linalg.norm(y_true)
        
        # 返回相对 L2 误差
        return error_norm / (true_norm + eps)

    # 3. 计算三个分量的 L2 误差
    err_u = safe_rel_l2(u_true_flat, u_pred_flat)
    err_v = safe_rel_l2(v_true_flat, v_pred_flat)
    err_vel = safe_rel_l2(vel_true_flat, vel_pred_flat)
    
    return err_u, err_v, err_vel

def plot_solution(model, epoch):
    """画 u, v, velocity 的对比图"""
    X_tensor = torch.tensor(X_star, dtype=torch.float32, device=device)
    X_tensor.requires_grad_(True)
    
    model.eval()
    uhat = model(X_tensor)
    u_pred, v_pred, _ = NS_uv(uhat, X_tensor)
    
    u_pred = u_pred.detach().cpu().numpy().reshape(nx, ny)
    v_pred = v_pred.detach().cpu().numpy().reshape(nx, ny)
    vel_pred = np.sqrt(u_pred**2 + v_pred**2)
    
    # 真解 reshape
    u_true_grid = u_sol.reshape(nx, ny)
    v_true_grid = v_sol.reshape(nx, ny)
    vel_true_grid = velocity_sol.reshape(nx, ny)
    
    os.makedirs('solutions_ns', exist_ok=True)
    
    # 画 Velocity Magnitude
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].pcolormesh(X, Y, vel_true_grid, shading='auto', cmap='jet')
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("True Velocity")
    
    im1 = axes[1].pcolormesh(X, Y, vel_pred, shading='auto', cmap='jet')
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title(f"Predicted Velocity (Epoch {epoch})")
    
    err = np.abs(vel_true_grid - vel_pred)
    im2 = axes[2].pcolormesh(X, Y, err, shading='auto', cmap='jet')
    fig.colorbar(im2, ax=axes[2])
    axes[2].set_title("Absolute Error")
    
    plt.tight_layout()
    plt.savefig(f'solutions_ns/epoch_{epoch}.png')
    plt.close()

def plot_residual_and_samples(model, data, epoch):
  
    # 构造绘图网格
    n_plot = 100
    x_p = np.linspace(lx, rx, n_plot)
    y_p = np.linspace(ly, ry, n_plot)
    X_p, Y_p = np.meshgrid(x_p, y_p)
    X_star_p = np.hstack((X_p.flatten()[:, None], Y_p.flatten()[:, None]))
    X_tensor = torch.tensor(X_star_p, dtype=torch.float32, device=device)
    X_tensor.requires_grad_(True)
    
    model.eval()
    # 计算全场残差
    with torch.set_grad_enabled(True):
        uhat = model(X_tensor)
        f_u, f_v = NS_res(uhat, X_tensor)
        # 欧拉残差模长
        res_val = (f_u**2 + f_v**2).detach().cpu().numpy().reshape(n_plot, n_plot)
        
    samples = data.detach().cpu().numpy()
    
    os.makedirs('residual_samples_ns', exist_ok=True)
    
    # [修复] 恢复成 1x3 的图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmax = np.percentile(res_val, 98) # 动态调整 colorbar 范围
    
    # 1. Residual Heatmap
    im0 = axes[0].pcolormesh(X_p, Y_p, res_val, shading='auto', cmap='jet', vmax=vmax)
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title(f"Residual (Epoch {epoch})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    
    # 2. Residual + Samples Overlay
    im1 = axes[1].pcolormesh(X_p, Y_p, res_val, shading='auto', cmap='jet', vmax=vmax)
    plt.colorbar(im1, ax=axes[1])
    axes[1].scatter(samples[:,0], samples[:,1], s=1.8, c='white', alpha=0.5, label='Samples')
    axes[1].set_title("Residual + Samples")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_xlim(lx, rx); axes[1].set_ylim(ly, ry)
    
    # 3. [新增回来的] Pure Samples
    axes[2].scatter(samples[:,0], samples[:,1], s=1.8, c='red', alpha=0.6)
    axes[2].set_title(f"Sample Distribution (N={len(samples)})")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    axes[2].set_xlim(lx, rx); axes[2].set_ylim(ly, ry)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'residual_samples_ns/res_epoch_{epoch}.png')
    plt.close()

def plot_loss_history(histories):
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.semilogy(hist, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_history_ns.png')
    plt.close()

# ===========================
#       主训练循环
# ===========================
def train(bounds, method=1, guding_lr=False):
    
    print(f"Training Navier-Stokes (Lid Driven Cavity)")
    
    # 初始化采样
    X_f, X_walls, X_lid = get_samplers(Samples, N_f=num_r, N_b=num_b, T=20)
    plot_residual_and_samples(model, X_f, epoch=0)
    
    # 权重初始化
    lambd_walls = torch.tensor(1.0, device=device, requires_grad=True)
    lambd_lid   = torch.tensor(1.0, device=device, requires_grad=True)
    
    params = [{'params': model.parameters(), 'lr': lr}]
    if guding_lr:
        optimizer = Adam(params, betas=(0.99, 0.999))
        scheduler = None
    else:
        optimizer = Adam(params)
        scheduler = MultiStepLR(optimizer, milestones=[10000, 20000], gamma=0.1)
        
    # 记录 Loss
    histories = {
        'total': [], 'pde': [], 'walls': [], 'lid': []
    }
    
    # 移动平均
    mean_loss = {'pde': 0.0, 'walls': 0.0, 'lid': 0.0}
    
    for epoch in range(Adam_n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 确保梯度开启
        if not X_f.requires_grad: X_f.requires_grad_(True)
        
        # IDEAL 重采样
        if epoch % N_change_point == 0 and epoch != 0:
            X_f, _, _ = get_samplers(Samples, N_f=num_r, N_b=num_b, T=10)
            plot_residual_and_samples(model, X_f, epoch)
            plot_solution(model, epoch)
            
        # 前向计算
        uhat = model(X_f)
        f_u, f_v = NS_res(uhat, X_f)
        loss_pde = torch.mean(f_u**2 + f_v**2)
        
        # Walls BC (u=0, v=0)
        pred_walls = model(X_walls)
        u_w, v_w, _ = NS_uv(pred_walls, X_walls)
        loss_walls = torch.mean(u_w**2 + v_w**2)
        
        # Lid BC (u=1, v=0)
        pred_lid = model(X_lid)
        u_l, v_l, _ = NS_uv(pred_lid, X_lid)
        loss_lid = torch.mean((u_l - 1.0)**2 + v_l**2)
        
        # 记录
        histories['pde'].append(loss_pde.item())
        histories['walls'].append(loss_walls.item())
        histories['lid'].append(loss_lid.item())
        
        # 动态权重更新 (仿照 File 2 的逻辑)
        # 这里简化实现 DB-PINN 的 mean 策略作为示例
        if method == 1 and epoch % N_change_lambda == 0 and epoch != 0:
            # 更新移动平均
            beta = 0.9
            mean_loss['pde'] = beta * mean_loss['pde'] + (1-beta) * loss_pde.item()
            mean_loss['walls'] = beta * mean_loss['walls'] + (1-beta) * loss_walls.item()
            mean_loss['lid'] = beta * mean_loss['lid'] + (1-beta) * loss_lid.item()
            
            # 计算梯度统计
            max_r, mean_r = loss_grad_max_mean(loss_pde, model)
            max_w, mean_w = loss_grad_max_mean(loss_walls, model, lambg=lambd_walls)
            max_l, mean_l = loss_grad_max_mean(loss_lid, model, lambg=lambd_lid)
            
            # 权重更新公式 (参考 File 2)
            hat_w = max_r / mean_w
            hat_l = max_r / mean_l
            
            # 简单的平滑更新
            with torch.no_grad():
                lambd_walls.data = 0.9 * lambd_walls + 0.1 * hat_w
                lambd_lid.data   = 0.9 * lambd_lid + 0.1 * hat_l
        
        # 总 Loss
        loss = loss_pde + lambd_walls * loss_walls + lambd_lid * loss_lid
        histories['total'].append(loss.item())
        
        loss.backward()
        
        if guding_lr:
            optimizer.step()
        else:
            optimizer.step()
            scheduler.step()
            
        # 打印与绘图
        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            err_u, err_v, err_vel = compute_L2_error(model)
            print(f"Ep {epoch}: Loss {loss.item():.4e} (PDE {loss_pde:.4e}) | "
                  f"L2 Vel {err_vel:.4e} | Weights: W {lambd_walls.item():.2f}, L {lambd_lid.item():.2f}")
                  
        
            
            
    plot_loss_history(histories)
    return model

if __name__ == "__main__":
    train(bounds=bounds, method=1)