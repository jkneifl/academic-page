# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import imageio
from scipy.stats.qmc import LatinHypercube
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader
# local imports
import CROMAutoencoder
import utils

"""
Diffusion equation
    u_t - alpha * u_xx = 0

"""

def get_alpha(alpha1, alpha2, alpha3):
    alpha = np.ones((Nx+1))  # Diffusion coefficient
    Na = int((Nx + 1) / 3)
    alpha[:Na], alpha[Na:-Na], alpha[-Na:] = alpha1, alpha2, alpha3
    return alpha

def solve_diffusion(Nt, Nx, dx, dt, alpha, u_init):
    """
    Solve the diffusion equation u_t - alpha * u_xx = 0 with a first-order Euler scheme
    :param Nt: number of time steps
    :param Nx: number of spatial grid points
    :param dx: spatial step size
    :param dt: time step size
    :param alpha: diffusion coefficient
    :param u_init: initial condition
    :return: u
    """
    # create temperature vector of size (timesteps x (spatial points + 1))
    u = np.zeros((Nt, Nx + 1))
    # set initial condition
    u[0] = u_init

    # Main time-stepping loop
    for t in range(1, Nt):
        # Compute the second derivative using finite differences
        u_xx = np.zeros(Nx + 1)
        u_xx[1:Nx] = (u[t - 1, 2:] - 2 * u[t - 1, 1:-1] + u[t - 1, :-2]) / (dx ** 2)
        # Update the solution
        u_t = alpha * u_xx
        # update solution using a first-order Euler scheme
        u[t] = u[t - 1] + dt * u_t

    return u

# Parameters
train_model = False
r = 16  # latent dimensio
L = 50.0  # Length of the domain
T = 0.00025  # Total simulation time
Nx = 500  # Number of spatial grid points
Nt = 150  # Number of time steps
Nsim = 30  # Number of simulations
Na = int((Nx + 1) / 3)
x = np.linspace(0, L, Nx + 1)  # Mesh points in space
t = np.linspace(0, T, Nt + 1)  # Mesh points in time

# Spatial and time step sizes
dx = L / Nx
dt = T / Nt

# Initial condition
with h5py.File("/Users/jonaskneifl/Develop/30_Lectures/01_CROM/CROM_online_deployment/data/Diffusion/h5_f_0000000000.h5", "r") as f:
    print("Keys: %s" % f.keys())
    u_init = f['q'][()].squeeze()
    x2 = f['x'][()]

# create data vector
u = np.zeros((Nsim, Nt, Nx + 1))

# sample different diffusion parameters using lhs
sampler = LatinHypercube(d=3, seed=0)
samples = 2000*sampler.random(n=Nsim)
for i in range(Nsim):
    alpha1, alpha2, alpha3 = samples[i]
    alpha = get_alpha(alpha1, alpha2, alpha3)
    u[i] = solve_diffusion(Nt, Nx, dx, dt, alpha, u_init)

# # Plot the results in subplot 1 x 2
i_sim = 15
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
ax[0].plot(x, u[i_sim, 0], label='t=0')
plt.ylabel('u')
plt.xlabel('x')
ax[1].plot(x, u[i_sim, -1], label='t=T/2')
plt.xlabel('x')
plt.legend()
plt.show()

# create gif of diffusion process
utils.diffusion_gif(x, u, samples, t, Nt, i_sim=i_sim, output_dir = "")


# %% train autoencoder on diffusion data
Ntrain = 20
u_train = u[:Ntrain].reshape(-1, Nx+1)
x_train = np.repeat(x[np.newaxis], Ntrain*Nt, axis=0)
u_test = u[Ntrain:].reshape(-1, Nx+1)
x_test = np.repeat(x[np.newaxis], (Nsim - Ntrain)*Nt, axis=0)
samples_test = samples[Ntrain:].reshape(-1, 3)

# create dataset
dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(u_train).float())
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# create model
model = CROMAutoencoder(r=r, Nx=u_train.shape[1], Nd=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# train model
if train_model:
    num_epochs = 12000
    losses = []
    for epoch in range(num_epochs):
        for data in train_loader:
            x, u = data
            # ===================forward=====================
            output = model(x, u)
            loss = criterion(output, u)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.6f}'
              .format(epoch + 1, num_epochs, loss.item()))
        losses.append(loss.item())

    # save model
    torch.save(model.state_dict(), f'model.pth')
    # plot loss
    plt.semilogy(losses)
else:
    model.load_state_dict(torch.load(f'model.pth'))


# %% test autoencoder

# reconstruct data
u_rec_test = model(torch.from_numpy(x_test).float(), torch.from_numpy(u_test).float()).detach().numpy()
# # create gif of reconstruction
utils.reconstruction_gif(x, u, u_rec_test, samples, t, Nt, i_sims=[i_sim], output_dir="")

# %% latent dynamics
i_test = 0
support_point_indices = [51, 205, 315, 261, 214, 144, 356, 267, 115, 271, 244, 309,  62, 364,
        412, 467, 249, 403, 175, 328, 105, 354]
# support_point_indices = range(Nx+1)
Nsupport = len(support_point_indices)
x_support = torch.from_numpy(x[support_point_indices]).float().requires_grad_().view([1, -1])

forward = model.decoder
def time_stepping(x_support, alpha_support, u_init):
    # encode initial condition
    z_init = model.encoder(torch.from_numpy(u_init).float()).detach().requires_grad_()
    # create latent vector of size (timesteps x (spatial points + 1))
    z = torch.zeros((Nt, r))
    # set initial condition
    z[0] = z_init
    # Main time-stepping loop
    for i_time in range(1, Nt):
        u_old = forward(x_support, z[i_time-1:i_time])
        # compute the second order spatial gradient using autograd
        hessian = torch.func.hessian(forward, argnums=0)(x_support, z[i_time-1:i_time]).squeeze()
        u_xx = torch.diagonal(torch.diagonal(hessian, dim1=0, dim2=1), dim1=0, dim2=1).detach().squeeze()
        # apply boundary conditions
        for i, index in enumerate(support_point_indices):
            if index == 0 or index == Nx:
                u_xx[i] = 0

        # get time derivative
        u_t = alpha_support * u_xx

        # evolve latent variable in time (this part uses the linearized version instead of Gauss-Newton solver)
        res = u_t
        jac = torch.func.jacrev(forward, argnums=1)(x_support, z[i_time-1:i_time]).detach().squeeze()
        vhat = torch.inverse(torch.matmul(jac.transpose(1, 0), jac)).matmul(jac.transpose(1, 0)).matmul(res)
        vhat = vhat.view(1, 1, -1)  # (dim: ? x ? x r)
        z[i_time] = z[i_time-1] + vhat * dt

    return z

# approximate full vector field
# lets reduce the size in order not to blow out colab. Hessians require significant memory
alpha = get_alpha(samples_test[i_test, 0], samples_test[i_test, 1], samples_test[i_test, 2])
alpha_support = torch.from_numpy(alpha[support_point_indices]).float()
z = time_stepping(x_support, alpha_support, u_init)
u_rec_test = model.decoder(torch.from_numpy(x_test[:150]).float(), z).detach().numpy()

# %% Approximation solution

