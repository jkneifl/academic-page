import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt

def solve_diffusion(n_t, n_x, dx, dt, alpha, u_init):
    """
    Solve the diffusion equation u_t - alpha * u_xx = 0 with a first-order Euler scheme
    :param n_t: number of time steps
    :param n_x: number of spatial grid points
    :param dx: spatial step size
    :param dt: time step size
    :param alpha: diffusion coefficient
    :param u_init: initial condition
    :return: u
    """
    # create temperature vector of size (timesteps x (spatial points + 1))
    u = np.zeros((n_t, n_x + 1))
    # set initial condition
    u[0] = u_init

    # Main time-stepping loop
    for t in range(1, n_t):
        # Compute the second derivative using finite differences
        u_xx = np.zeros(n_x + 1)
        u_xx[1:n_x] = (u[t - 1, 2:] - 2 * u[t - 1, 1:-1] + u[t - 1, :-2]) / (dx ** 2)
        # Update the solution
        u_t = alpha * u_xx
        # update solution using a first-order Euler scheme
        u[t] = u[t - 1] + dt * u_t

    return u

def get_alpha(n_x, alpha1, alpha2, alpha3):
    """
    Get spatially dependent diffusion coefficient that is split into three constant intervals
    :param n_x: number of spatial grid points
    :param alpha1: diffusion coefficient in the first interval
    :param alpha2: diffusion coefficient in the second interval
    :param alpha3: diffusion coefficient in the third interval
    :return:
    """
    alpha = np.ones((n_x+1))  # Diffusion coefficient
    Na = int((n_x + 1) / 3)
    alpha[:Na], alpha[Na:-Na], alpha[-Na:] = alpha1, alpha2, alpha3
    return alpha

def time_stepping(model, n_t, n_x, dt, x_support, support_point_indices, alpha_support, u_init):
    """
    Time evolution of the latent variable by evaluating the original PDE on a small number of support points
    and then finding the latent variables that matches best to the found evolved vector field values
    :param model: CROMAutoencoder model
    :param n_t: number of time steps
    :param n_x: number of spatial grid points
    :param dt: time step size
    :param x_support: spatial grid points that are evaluated during time stepping
    :param support_point_indices: indices of the support points
    :param alpha_support: diffusion coefficients at the support points
    :param u_init: initial condition of the vector field
    :return:
    """
    # encode initial condition
    z_init = model.encoder(torch.from_numpy(u_init).float()).detach().requires_grad_()
    # create latent vector of size (timesteps x (spatial points + 1))
    z = torch.zeros((n_t, model.r))
    # set initial condition
    z[0] = z_init
    # Main time-stepping loop
    for i_time in range(1, n_t):
        # compute the second order spatial gradient using autograd
        hessian = torch.func.hessian(model.decoder, argnums=0)(x_support, z[i_time-1:i_time]).squeeze()
        u_xx = torch.diagonal(torch.diagonal(hessian, dim1=0, dim2=1), dim1=0, dim2=1).detach().squeeze()
        # apply boundary conditions
        for i, index in enumerate(support_point_indices):
            if index == 0 or index == n_x:
                u_xx[i] = 0

        # get time derivative
        u_t = alpha_support * u_xx

        # evolve latent variable in time (this part uses the linearized version instead of Gauss-Newton solver)
        res = u_t
        jac = torch.func.jacrev(model.decoder, argnums=1)(x_support, z[i_time-1:i_time]).detach().squeeze()
        vhat = torch.inverse(torch.matmul(jac.transpose(1, 0), jac)).matmul(jac.transpose(1, 0)).matmul(res)
        vhat = vhat.view(1, 1, -1)  # (dim: ? x ? x r)
        z[i_time] = z[i_time-1] + vhat * dt

    return z

def diffusion_gif(x, u, alphas, t, n_t, n_x, i_sim=0, output_dir=""):
    """
    create gif of diffusion process
    :param x: spatial grid points
    :param u: vector field values
    :param alphas: diffusion coefficients
    :param t: time points
    :param n_t: number of time steps
    :param n_x: number of spatial grid points
    :param i_sim: simulation index
    :param output_dir: output directory
    """
    # figure without gui
    plt.ioff()
    images = []
    for i in range(n_t):
        fig = plt.figure(figsize=(8, 8*0.64))
        plt.plot(x, u[i_sim, i], 'm', label='t=T/2')
        images.append(format_figure(fig, t[i], x, alphas, n_x, i_sim))
    # # repeated gif
    imageio.mimsave(os.path.join(output_dir, 'gifs/diffusion.gif'), images, loop=1)
    plt.ion()

def reconstruction_gif(x, u, u_rec, alphas, t, n_t, n_x, i_sims=0, output_dir=""):
    """
    create gif of reconstruction vs FOM diffusion process
    :param x: spatial grid points
    :param u: vetor field values
    :param u_rec: reconstructed vector field values
    :param alphas: diffusion coefficients
    :param t: time points
    :param n_t: number of time steps
    :param n_x: number of spatial grid points
    :param i_sim: simulation index
    :param output_dir: output directory
    :return:
    """
    plt.ioff()
    for i_sim in i_sims:
        images = []
        for i in range(i_sim*n_t, (i_sim+1)*n_t):
            # figure without gui
            fig = plt.figure(figsize=(8, 8*0.64))
            plt.plot(x, u[i], 'm', label='FOM')
            plt.plot(x, u_rec[i], 'c', label='Reconstruction')
            images.append(format_figure(fig, t[i], x, alphas, n_x, i_sim))
        # repeated gif
        imageio.mimsave(os.path.join(output_dir, f'gifs/diffusion_rec_test_{i_sim+1}.gif'), images, loop=1)
    plt.ion()

def approximation_gif(x, u, u_approx, alphas, t, n_t, n_x, support_point_indices, i_sims=[0], output_dir=""):
    """
    create gif of approximation vs FOM diffusion process
    :param x: spatial grid points
    :param u: vector field values
    :param u_rec: reconstructed vector field values
    :param alphas: diffusion coefficients
    :param t: time points
    :param n_t: number of time steps
    :param n_x: number of spatial grid points
    :param support_point_indices: indices that are evaluated during time stepping
    :param i_sims: list of simulation indices
    :param output_dir: output directory
    """
    plt.ioff()
    # create gif of approximations
    for i_sim in i_sims:
        images = []
        for i in range(i_sim * n_t, (i_sim + 1) * n_t):
            # figure without gui
            fig = plt.figure(figsize=(8, 8 * 0.64))
            plt.plot(x, u[i], 'm', label='FOM')
            plt.plot(x, u_approx[i_sim][i % n_t], 'c', label='Approximation')
            plt.plot(x[support_point_indices], u_approx[i_sim][i % n_t, support_point_indices], 'co',
                     label='integration points')
            images.append(format_figure(fig, t[i], x, alphas, n_x, i_sim))
        # repeated gif
        imageio.mimsave(os.path.join(output_dir, f'gifs/diffusion_approx_{i_sim + 1}.gif'), images, loop=1)
    plt.ion()

def format_figure(fig, t, x, alphas, n_x, i_sim):
    """
    format figure for gif with diffusion factor intervals and correct labels, ticks, etc.
    """
    interval_length = int((n_x + 1) / 3)
    plt.axvspan(x[0], x[interval_length], facecolor='grey', alpha=alphas[i_sim, 0] / alphas.max())
    plt.axvspan(x[1 * interval_length], x[2 * interval_length], facecolor='grey', alpha=alphas[i_sim, 1] / alphas.max())
    plt.axvspan(x[2 * interval_length], x[-1], facecolor='grey', alpha=alphas[i_sim, 2] / alphas.max())
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.ylim([-1, 1])
    plt.xlim([0, x.max()])
    plt.title(f'$t={t * 1000:.3f}\,$ms')
    # increase font size
    plt.rcParams.update({'font.size': 16})
    # increase label and tick size
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    fig.savefig('gifs/tmp.png')
    # close figure
    plt.close(fig)
    image = imageio.imread('gifs/tmp.png')
    return image
