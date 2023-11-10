# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py


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
from scipy.stats.qmc import LatinHypercube
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
print('a')

# create gif of diffusion process
# import imageio
# output_dir = ""
# images = []
# for i in range(Nt):
#     # figure without gui
#     plt.ioff()
#     fig = plt.figure(figsize=(8, 8*0.64))
#     plt.plot(x, u[i_sim, i], 'm', label='t=T/2')
#     plt.axvspan(x[0], x[167], facecolor='grey', alpha=samples[i_sim, 0])
#     plt.axvspan(x[1*167], x[2*167], facecolor='grey', alpha=samples[i_sim, 1])
#     plt.axvspan(x[2*167], x[-1], facecolor='grey', alpha=samples[i_sim, 2])
#     plt.xlabel('$x$')
#     plt.ylabel('$u$')
#     plt.ylim([-1, 1])
#     plt.xlim([0, x.max()])
#     plt.title(f'$t={t[i]*1000:.3f}\,$ms')
#     # increase font size
#     plt.rcParams.update({'font.size': 16})
#     # increase label and tick size
#     plt.tick_params(axis='both', which='major', labelsize=16)
#     plt.tight_layout()
#     fig.savefig('gifs/tmp.png')
#     images.append(imageio.imread('gifs/tmp.png'))
# # # repeated gif
# imageio.mimsave('gifs/diffusion.gif', images, loop=1)


x_factor = 1
# %% train autoencoder on diffusion data
Ntrain = 20
u_train = u[:Ntrain].reshape(-1, Nx+1)
x_train = np.repeat(x[np.newaxis], Ntrain*Nt, axis=0)
u_test = u[Ntrain:].reshape(-1, Nx+1)
x_test = np.repeat(x[np.newaxis], (Nsim - Ntrain)*Nt, axis=0)
samples_test = samples[Ntrain:].reshape(-1, 3)


# %% PyTorch Implementation
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader
# create dataset
dataset = TensorDataset(torch.from_numpy(x_factor*x_train).float(), torch.from_numpy(u_train).float())
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# create autoencoder
class TorchAutoencoder(nn.Module):
    def __init__(self, r, Nx, Nd=1, n_layers=3, n_neurons=128):
        """
        :param r: latent space dimension
        :param Nx: number of spatial points
        :param Nd: number of coordinates required to describe the position (1 for 1D, 2 for 2D, ...)
        """
        super(TorchAutoencoder, self).__init__()
        self.r = r
        self.Nx = Nx
        self.Nd = Nd
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        # create layers for the encoder
        self.encoder_0 = nn.Linear(self.Nx, self.n_neurons)
        self.encoder_0_act = nn.ELU(True)
        for i in range(n_layers-2):
            setattr(self, f'encoder_{i+1}', nn.Linear(self.n_neurons, self.n_neurons))
            setattr(self, f'encoder_{i+1}_act', nn.ELU(True))
        setattr(self, f'encoder_{n_layers-1}', nn.Linear(self.n_neurons, self.r))

        # create layers for encoder
        self.decoder_0 = nn.Linear(self.r + self.Nd, self.n_neurons)
        self.decoder_0_act = nn.ELU(True)
        for i in range(n_layers-2):
            setattr(self, f'decoder_{i+1}', nn.Linear(self.n_neurons, self.n_neurons))
            setattr(self, f'decoder_{i+1}_act', nn.ELU(True))
        setattr(self, f'decoder_{n_layers-1}', nn.Linear(self.n_neurons+1, 1))

        # self.encoder = nn.Sequential(
        #     nn.Linear(self.Nx, 128),
        #     nn.ELU(True),
        #     nn.Linear(128, 64),
        #     nn.ELU(True),
        #     nn.Linear(64, self.r),
        #     nn.ELU(True),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.r + self.Nd, 128),
        #     # nn.ELU(True),
        #     # nn.Linear(self.r, 64),
        #     nn.ELU(True),
        #     nn.Linear(128, 128),
        #     nn.ELU(True),
        #     nn.Linear(128, 128),
        #     nn.ELU(True),
        #     nn.Linear(128, 1)
        # )
    def encoder(self, u):
        for i in range(self.n_layers):
            u = getattr(self, f'encoder_{i}')(u)
            if hasattr(self, f'encoder_{i}_act'):
                u = getattr(self, f'encoder_{i}_act')(u)
        return u

    def decoder(self, x, z):

        # save batch size for reshaping later
        batch_size_local = x.size(0)

        # repeat z for every point in x (batch_size_local x Nx x r)
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        # add new axis to x to the end (batch_size_local x Nx x Nd)
        x = x.unsqueeze(2)
        # concatenate x and z
        decoder_input = torch.cat((x, z), dim=2)
        # reshape for decoder so that all the points are processed at once (batchsize = batch_size_local * Nx)
        decoder_input = decoder_input.view(-1, self.r + self.Nd)

        u_ = decoder_input
        for i in range(self.n_layers-1):
            u_ = getattr(self, f'decoder_{i}')(u_)
            if hasattr(self, f'decoder_{i}_act'):
                u_ = getattr(self, f'decoder_{i}_act')(u_)
        # stack x and u_ (skip connection from x to output layer)
        output_input = torch.cat((x.view(-1, self.Nd), u_), dim=1)
        u_ = getattr(self, f'decoder_{self.n_layers-1}')(output_input)

        # reshape x_rec
        u_rec = u_.view(batch_size_local, -1)

        return u_rec

    def forward(self, x, u):
        """
        :param x: (batchsize x n) positional coordinates
        :param u: (batchsize x n) vector field values
        """

        # save batch size for reshaping later
        batch_size_local = x.size(0)

        # encode input to latent space
        z = self.encoder(u)

        # reshape x_rec
        u_rec = self.decoder(x, z)

        return u_rec

    def forward2(self, x, u):
        """
        :param x: (batchsize x n) positional coordinates
        :param u: (batchsize x n) vector field values
        """

        # save batch size for reshaping later
        batch_size_local = x.size(0)

        # encode input to latent space
        z = self.encoder(u)


        # repeat z for every point in x
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        # add new axis to x in the middle
        x = x.unsqueeze(2)
        # concatenate x and z
        decoder_input = torch.cat((x, z), dim=2)
        # reshape for decoder so that all the points are processed at once (batchsize = batch_size_local * Nx)
        decoder_input = decoder_input.view(-1, self.r + self.Nd)

        # decode latent space to output
        u_rec = self.decoder(decoder_input)
        # reshape x_rec
        u_rec = u_rec.view(batch_size_local, -1)

        return u_rec

# create model
model = TorchAutoencoder(r=r, Nx=u_train.shape[1], Nd=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# train model
model.load_state_dict(torch.load(f'model_{x_factor}.pth'))

if False:
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
    torch.save(model.state_dict(), f'model_{x_factor}.pth')
    # plot loss
    plt.semilogy(losses)
# load model
model.load_state_dict(torch.load(f'model_{x_factor}.pth'))


# %% test autoencoder

# reconstruct data
i_sample = 248
# u_rec = model(torch.from_numpy(x_factor*x_train).float(), torch.from_numpy(u_train).float()).detach().numpy()
# plt.plot(u_rec[i_sample])
# plt.plot(u_train[i_sample])
# plt.show()

# u_rec_test = model(torch.from_numpy(x_factor*x_test).float(), torch.from_numpy(u_test).float()).detach().numpy()
# # create gif of reconstruction
# import imageio
# output_dir = ""
# i_sim = 1
# for i_sim in range(10):
#     images = []
#     for i in range(i_sim*Nt, (i_sim+1)*Nt):
#         # figure without gui
#         plt.ioff()
#         # plot with gui
#
#         fig = plt.figure(figsize=(8, 8*0.64))
#         plt.plot(x_test[0], u_test[i], 'm', label='FOM')
#         plt.plot(x_test[0], u_rec_test[i], 'c', label='Reconstruction')
#         plt.axvspan(x_test[0, 0], x_test[0, 167], facecolor='grey', alpha=samples_test[i_sim, 0]/samples_test.max())
#         plt.axvspan(x_test[0, 1*167], x_test[0, 2*167], facecolor='grey', alpha=samples_test[i_sim, 1]/samples_test.max())
#         plt.axvspan(x_test[0, 2*167], x_test[0, -1], facecolor='grey', alpha=samples_test[i_sim, 2]/samples_test.max())
#         plt.xlabel('$x$')
#         plt.ylabel('$u$')
#         plt.ylim([-1, 1])
#         plt.xlim([0, x_test.max()])
#         # legend with fixed position upper right
#         plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
#         plt.title(f'$t={t[i%Nt]*1000:.3f}\,$ms')
#         # increase font size
#         plt.rcParams.update({'font.size': 16})
#         # increase label and tick size
#         plt.tick_params(axis='both', which='major', labelsize=16)
#         plt.tight_layout()
#         fig.savefig('gifs/tmp.png')
#         images.append(imageio.imread('gifs/tmp.png'))
#     # repeated gif
#     imageio.mimsave(f'gifs/diffusion_rec_test_{i_sim+1}.gif', images, loop=1)

# %% latent dynamics
i_test = 0
support_point_indices = [0, 51, 205, 315, 261, 214, 144, 356, 267, 115, 271, 244, 309,  62, 364,
        412, 467, 249, 403, 175, 328, 105, 354, 500]
# support_point_indices = range(Nx+1)
Nsupport = len(support_point_indices)

x_support = torch.from_numpy(x[support_point_indices]*x_factor).float().requires_grad_().view([1, -1])
# lets reduce the size in order not to blow out colab. Hessians require significant memory
alpha = get_alpha(samples_test[i_test, 0], samples_test[i_test, 1], samples_test[i_test, 2])
alpha_support = torch.from_numpy(alpha[support_point_indices]).float()

def forward(x, z):
    """
    :param x: set of positional coordinates
    :param z: latent space coordinates
    """

    n = x.size(0)
    # repeat z for every point in x
    z = z.unsqueeze(0).repeat(n, 1)
    # add new axis to x in the middle
    x = x.unsqueeze(1)
    # concatenate x and z
    decoder_input = torch.cat((x, z), dim=1)
    # cast to float
    decoder_input = decoder_input.float()

    # decode latent space to output
    u_n = model.decoder(decoder_input).squeeze()

    return u_n

forward = model.decoder

# encode initial condition
z_init = model.encoder(torch.from_numpy(u_init).float()).detach().requires_grad_()
# create latent vector of size (timesteps x (spatial points + 1))
z = torch.zeros((Nt, r))
# set initial condition
z[0] = z_init

from torch.autograd import grad
def nth_derivative(f, wrt, n):

    for i in range(n):

        grads = grad(f, wrt, create_graph=True, allow_unused=True)[0]
        f = grads.sum()

    return grads

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
    # update solution using a first-order Euler scheme
    # u_new = u_old + dt * u_t
    u_new = u_old + u_t

    # evolve latent variable in time (this part uses the lineriazed version instead of Gauss-Newton solver)
    res = u_t
    jac = torch.func.jacrev(forward, argnums=1)(x_support, z[i_time-1:i_time]).detach().squeeze()
    vhat = torch.inverse(torch.matmul(jac.transpose(1, 0), jac)).matmul(jac.transpose(1, 0)).matmul(res)
    vhat = vhat.view(1, 1, -1)  # (dim: ? x ? x r)
    z[i_time] = z[i_time-1] + vhat * dt

    # # find z_new that matches u_new the best
    # z[t] = find_best_match(u_new, x_support, z[t-1])

# reconstruct full vector field

u_rec_start = forward(torch.from_numpy(x_factor*x).float().view([1, -1]), z[0:1]).detach().numpy()
u_rec_end = forward(torch.from_numpy(x_factor*x).float().view([1, -1]), z[-1:]).detach().numpy()

plt.plot(u_rec_start.T)
plt.plot(u_rec_end.T, "--")
plt.legend(['start', 'end'])
plt.show()

# %% Approximation solution

u_rec_test = model.decoder(torch.from_numpy(x_factor*x_test[:150]).float(), z).detach().numpy()
# create gif of reconstruction
import imageio
output_dir = ""
for i_test in range(10):
    images = []
    for i in range(i_test*Nt, (i_test+1)*Nt):
        # figure without gui
        plt.ioff()
        # plot with gui

        fig = plt.figure(figsize=(8, 8*0.64))
        plt.plot(x_test[0], u_test[i], 'm', label='FOM')
        plt.plot(x_test[0], u_rec_test[i], 'c', label='Approximation')
        plt.plot(x_test[0, support_point_indices], u_rec_test[i, support_point_indices], 'co', label='integration points')
        plt.axvspan(x_test[0, 0], x_test[0, 167], facecolor='grey', alpha=samples_test[i_test, 0]/samples_test.max())
        plt.axvspan(x_test[0, 1*167], x_test[0, 2*167], facecolor='grey', alpha=samples_test[i_test, 1]/samples_test.max())
        plt.axvspan(x_test[0, 2*167], x_test[0, -1], facecolor='grey', alpha=samples_test[i_test, 2]/samples_test.max())
        plt.xlabel('$x$')
        plt.ylabel('$u$')
        plt.ylim([-1, 1])
        plt.xlim([0, x_test.max()])
        # legend with fixed position upper right
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        plt.title(f'$t={t[i%Nt]*1000:.3f}\,$ms')
        # increase font size
        plt.rcParams.update({'font.size': 16})
        # increase label and tick size
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        fig.savefig('gifs/tmp.png')
        images.append(imageio.imread('gifs/tmp.png'))
    # repeated gif
    imageio.mimsave(f'gifs/diffusion_approx_test_{i_test+1}.gif', images, loop=1)



plt.plot(z[:, 0])
plt.show()

u_init
u_xx2 = (u_init[2:] - 2 * u_init[1:-1] + u_init[:-2]) / (dx ** 2)
