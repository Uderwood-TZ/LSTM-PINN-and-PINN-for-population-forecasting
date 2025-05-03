import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
a_max, t_min, t_max = 100.0, 2024.0, 2054.0
t_span = t_max - t_min
P_scale = 9.54e6
alpha = t_span / a_max
def mu_interp(a):
    a = np.clip(a, 0, a_max)
    mu0, B = 0.006805083, 0.0003
    linear = mu0 + B * a
    mu60   = mu0 + B * 60
    return np.where(a < 60, linear, mu60 * np.exp(0.06 * (a - 60)))

def base_asfr(a):
    return 0.0022 * (a - 20) * (35 - a) * ((a >= 20) & (a <= 35))

def P0_interp(a):
    return P_scale * np.exp(-0.02 * a) * (1 + 0.3 * np.sin(0.2 * a))
mu_nd = lambda a_nd: torch.from_numpy(
    mu_interp(a_nd.detach().numpy() * a_max) * t_span
).float()
b_interp_placeholder = None
b_dim = lambda a_nd, t_nd: torch.from_numpy(
    b_interp_placeholder(
        a_nd.detach().numpy() * a_max,
        t_nd.detach().numpy() * t_span + t_min
    )
).float()

P0_nd = lambda a_nd: torch.from_numpy(
    P0_interp(a_nd.detach().numpy() * a_max) / P_scale
).float()
class PopulationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,64),  nn.Tanh(),
            nn.Linear(64,1)
        )
    def forward(self, a_nd, t_nd):
        x = torch.cat([a_nd, t_nd], dim=1)
        return self.fc(x) * (1 - a_nd)
def sample_points():
    a_int = torch.rand(N_int,1) * a_max
    t_int = torch.rand(N_int,1) * t_span + t_min
    a_ic  = torch.rand(N_ic,1) * a_max
    t_ic  = torch.full_like(a_ic, t_min)
    a_bc  = torch.zeros(N_bc,1)
    t_bc  = torch.rand(N_bc,1) * t_span + t_min

    a_int_nd = a_int / a_max
    t_int_nd = (t_int - t_min) / t_span
    a_ic_nd  = a_ic  / a_max
    t_ic_nd  = torch.zeros_like(a_ic_nd)
    a_bc_nd  = torch.zeros_like(a_bc)
    t_bc_nd  = (t_bc - t_min) / t_span

    return (a_int_nd.requires_grad_(True), t_int_nd.requires_grad_(True),
            a_ic_nd, t_ic_nd, a_bc_nd, t_bc_nd,
            a_int, t_int, a_ic, a_bc, t_bc)
def compute_loss(model, a_nd, t_nd, a_ic_nd, t_ic_nd, a_bc_nd, t_bc_nd,
                 a_int, t_int, a_ic, a_bc, t_bc):
    P_nd    = model(a_nd, t_nd)
    dPda    = autograd.grad(P_nd, a_nd, torch.ones_like(P_nd), create_graph=True)[0]
    dPdt    = autograd.grad(P_nd, t_nd, torch.ones_like(P_nd), create_graph=True)[0]
    res_l   = torch.mean((dPdt + alpha*dPda + mu_nd(a_nd)*P_nd)**2)

    P_ic    = model(a_ic_nd, t_ic_nd)
    P0v     = P0_nd(a_ic_nd)
    ic_l    = torch.mean(((P_ic - P0v)/(P0v + 1e-8))**2)

    a_mc    = torch.rand(N_bc,1)
    integr  = b_dim(a_mc, t_bc_nd) * model(a_mc, t_bc_nd)
    births  = a_max * integr.mean(dim=0)
    bc_l    = torch.mean((model(a_bc_nd, t_bc_nd) - births)**2)

    total_l = res_l + ic_l + bc_l
    return total_l, res_l, ic_l, bc_l
epochs = 10000
lr     = 5e-4
N_int  = 5000
N_ic   = 2000
N_bc   = 2000
scenarios = {
    'Three-child policy': lambda a, t: np.clip(
        base_asfr(a) * (1.0 + 0.2*(t>=2014) + 0.2*(t>=2016) + 0.2*(t>=2021)),
        0, 0.25
    ),
    'Two-child policy': lambda a, t: np.clip(
        base_asfr(a) * (1.0 + 0.2*(t>=2024.0)),
        0, 0.20
    ),
    'Universal two-child policy': lambda a, t: np.clip(
        base_asfr(a) * (1.0 + 0.2*(t>=2024.0)),
        0, 0.25
    ),
}

results = {}
loss_histories = {}
durations = {}
for name, b_fn in scenarios.items():
    print(f"\n=== Training scenarios: {name} ===")
    b_interp_placeholder = b_fn

    model = PopulationNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_histories[name] = {'total': [], 'pde': [], 'ic': [], 'bc': []}

    start_time = time.time()
    for epoch in range(epochs):
        pts = sample_points()
        total_l, r_l, ic_l, bc_l = compute_loss(model, *pts)
        optimizer.zero_grad()
        total_l.backward()
        optimizer.step()

        loss_histories[name]['total'].append(total_l.item())
        loss_histories[name]['pde'].append(r_l.item())
        loss_histories[name]['ic'].append(ic_l.item())
        loss_histories[name]['bc'].append(bc_l.item())

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:5d} — "
                f"Total: {total_l:.2e} | "
                f"PDE:   {r_l:.2e} | "
                f"IC:    {ic_l:.2e} | "
                f"BC:    {bc_l:.2e}"
            )
    durations[name] = time.time() - start_time
    # 预测并存储结果
    years = np.linspace(t_min, t_max, 30)
    age_grid = np.linspace(0, a_max, 100)
    pred = []
    for y in years:
        for a in age_grid:
            a_nd = torch.tensor([[a/a_max]], dtype=torch.float32)
            t_nd = torch.tensor([[(y-t_min)/t_span]], dtype=torch.float32)
            pred.append(model(a_nd, t_nd).item() * P_scale)
    results[name] = np.array(pred).reshape(100, 30)
for name, data in results.items():
    plt.figure(figsize=(6, 5))
    plt.contourf(years, age_grid, data, levels=50)
    plt.title(f'{name} Population projections for 2024 -- 2054')
    plt.xlabel('Year')
    plt.ylabel('Age')
    plt.colorbar(label='Population Density')
    plt.show()
for name, hist in loss_histories.items():
    plt.figure(figsize=(10, 6))
    plt.plot(hist['total'], label='Total Loss')
    plt.plot(hist['pde'],   label='PDE Loss')
    plt.plot(hist['ic'],    label='IC Loss')
    plt.plot(hist['bc'],    label='BC Loss')
    plt.title(f'{name} Training loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
with open('training_report.txt', 'w', encoding='utf-8') as f:
    for name in scenarios:
        f.write(f"=== Scene：{name} ===\n")
        f.write(f"Training duration：{durations[name]:.2f} s\n\n")

        f.write("—— Two-dimensional prediction results ——\n")
        years = np.linspace(t_min, t_max, 30)
        age_grid = np.linspace(0, a_max, 100)
        data = results[name]
        for i, age in enumerate(age_grid):
            for j, year in enumerate(years):
                f.write(f"Year {year:.1f}, Age {age:.1f} -> P = {data[i,j]:.3e}\n")
        f.write("\n")

        f.write("—— Loss history (epoch, total, PDE, IC, BC) ——\n")
        tot = loss_histories[name]['total']
        pde = loss_histories[name]['pde']
        ic  = loss_histories[name]['ic']
        bc  = loss_histories[name]['bc']
        for epoch in range(len(tot)):
            f.write(
                f"{epoch:5d}  "
                f"{tot[epoch]:.3e}  "
                f"{pde[epoch]:.3e}  "
                f"{ic[epoch]:.3e}  "
                f"{bc[epoch]:.3e}\n"
            )
        f.write("\n\n")
print("training_report.txt is generated")


























