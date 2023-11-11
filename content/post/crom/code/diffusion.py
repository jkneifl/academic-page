# third party imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy.stats.qmc import LatinHypercube
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
# local imports
from CROMAutoencoder import CROMAutoencoder
import utils

# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# %% Script parameters
""" Diffusion equation u_t - alpha * u_xx = 0"""

script_path = os.path.dirname(os.path.realpath(__file__))
# Parameters
train_model = False
create_gifs = False
# model parameters
r = 16  # latent dimension
L = 50.0  # Length of the spatial domain
T = 0.00025  # Total simulation time
n_x = 500  # Number of spatial grid points
n_t = 150  # Number of time steps
n_sims = 30  # Number of simulations
n_train = 20 # Number of training simulations
x = np.linspace(0, L, n_x + 1)  # Mesh points in space
t = np.linspace(0, T, n_t + 1)  # Mesh points in time
# Spatial and time step sizes
dx = L / n_x
dt = T / n_t

# load Initial condition
with h5py.File(os.path.join(script_path, "diffusion_initial_condition.h5"), "r") as f:
    u_init = f['q'][()].squeeze()

# create data vector
u = np.zeros((n_sims, n_t, n_x + 1))

# sample different diffusion parameters using lhs
sampler = LatinHypercube(d=3, seed=0)
diffusion_samples = 2000 * sampler.random(n=n_sims)

# %% Solve diffusion equation for different diffusion coefficients
for i in range(n_sims):
    alpha1, alpha2, alpha3 = diffusion_samples[i]
    alpha = utils.get_alpha(n_x, alpha1, alpha2, alpha3)
    u[i] = utils.solve_diffusion(n_t, n_x, dx, dt, alpha, u_init)

# create gif of diffusion process for example simulation
if create_gifs:
    utils.diffusion_gif(x, u, diffusion_samples, t, n_t, n_x, i_sim=0, output_dir ="")

# %% train autoencoder on diffusion data
u_train = u[:n_train].reshape(-1, n_x+1)
x_train = np.repeat(x[np.newaxis], n_train*n_t, axis=0)
u_test = u[n_train:].reshape(-1, n_x+1)
x_test = np.repeat(x[np.newaxis], (n_sims - n_train)*n_t, axis=0)
diffusion_samples_test = diffusion_samples[n_train:].reshape(-1, 3)

# create dataset
dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(u_train).float())
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# create model
model = CROMAutoencoder(r=r, n_x=u_train.shape[1], n_d=1)

# train model
if train_model:
    losses = model.fit(train_loader, num_epochs=12000)
    # plot loss
    plt.semilogy(losses)
else:
    model.load_state_dict(torch.load(f'model.pth'))


# %% test autoencoder

# reconstruct data
u_rec_test = model(torch.from_numpy(x_test).float(), torch.from_numpy(u_test).float()).detach().numpy()
# # create gif of reconstruction
if create_gifs:
    utils.reconstruction_gif(x, u_test, u_rec_test, diffusion_samples, t, n_t, n_x, i_sims=[0], output_dir="")

# %% latent dynamics

# indices of spatial points that are used for time stepping
support_point_indices = [51, 205, 315, 261, 214, 144, 356, 267, 115, 271, 244, 309,  62, 364,
        412, 467, 249, 403, 175, 328, 105, 354]
# support_point_indices = range(n_x+1)
Nsupport = len(support_point_indices)
x_support = torch.from_numpy(x[support_point_indices]).float().requires_grad_().view([1, -1])

# evolve latent variable in time for various diffusion coefficients
u_approx = []
for i_test in range(10):
    # get alpha for test simulation
    alpha = utils.get_alpha(n_x, diffusion_samples_test[i_test, 0], diffusion_samples_test[i_test, 1], diffusion_samples_test[i_test, 2])
    alpha_support = torch.from_numpy(alpha[support_point_indices]).float()
    # evolve latent variable in time
    z = utils.time_stepping(model, n_t, n_x, dt, x_support, support_point_indices, alpha_support, u_init)
    # reconstruct full vector field
    u_approx.append(model.decoder(torch.from_numpy(x_test[:n_t]).float(), z).detach().numpy())

# create gifs of approximation
# if create_gifs:
utils.approximation_gif(x, u_test, u_approx, diffusion_samples_test, t, n_t, n_x, support_point_indices,
                        i_sims=[0], output_dir="")
