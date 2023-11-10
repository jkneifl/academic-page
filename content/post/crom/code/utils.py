import os
import imageio
import torch
import matplotlib.pyplot as plt

def diffusion_gif(x, u, alphas, t, Nt, i_sim=0, output_dir=""):
    """
    create gif of diffusion process
    """

    images = []
    for i in range(Nt):
        # figure without gui
        plt.ioff()
        fig = plt.figure(figsize=(8, 8*0.64))
        plt.plot(x, u[i_sim, i], 'm', label='t=T/2')
        plt.axvspan(x[0], x[167], facecolor='grey', alpha=alphas[i_sim, 0])
        plt.axvspan(x[1*167], x[2*167], facecolor='grey', alpha=alphas[i_sim, 1])
        plt.axvspan(x[2*167], x[-1], facecolor='grey', alpha=alphas[i_sim, 2])
        plt.xlabel('$x$')
        plt.ylabel('$u$')
        plt.ylim([-1, 1])
        plt.xlim([0, x.max()])
        plt.title(f'$t={t[i]*1000:.3f}\,$ms')
        # increase font size
        plt.rcParams.update({'font.size': 16})
        # increase label and tick size
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        fig.savefig('gifs/tmp.png')
        images.append(imageio.imread('gifs/tmp.png'))
    # # repeated gif
    imageio.mimsave(os.path.join(output_dir, 'gifs/diffusion.gif'), images, loop=1)


def reconstruction_gif(x, u, u_rec, alphas, t, Nt, i_sims=0, output_dir=""):
    for i_sim in i_sims:
        images = []
        for i in range(i_sim*Nt, (i_sim+1)*Nt):
            # figure without gui
            plt.ioff()
            # plot with gui

            fig = plt.figure(figsize=(8, 8*0.64))
            plt.plot(x[0], u[i], 'm', label='FOM')
            plt.plot(x[0], u_rec[i], 'c', label='Reconstruction')
            plt.axvspan(x[0, 0], x[0, 167], facecolor='grey', alpha=alphas[i_sim, 0]/alphas.max())
            plt.axvspan(x[0, 1*167], x[0, 2*167], facecolor='grey', alpha=alphas[i_sim, 1]/alphas.max())
            plt.axvspan(x[0, 2*167], x[0, -1], facecolor='grey', alpha=alphas[i_sim, 2]/alphas.max())
            plt.xlabel('$x$')
            plt.ylabel('$u$')
            plt.ylim([-1, 1])
            plt.xlim([0, x.max()])
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
        imageio.mimsave(os.path.join(output_dir, f'gifs/diffusion_rec_test_{i_sim+1}.gif'), images, loop=1)


def approximation_gif(model, x, u, u_rec, x_support, alphas, t, Nt, support_point_indices, i_sims=0, output_dir=""):
    # create gif of reconstruction
    output_dir = ""
    u_init = u[0]
    for i_test in range(10):

        # lets reduce the size in order not to blow out colab. Hessians require significant memory
        alpha = get_alpha(alphas[i_test, 0], alphas[i_test, 1], alphas[i_test, 2])
        alpha_support = torch.from_numpy(alpha[support_point_indices]).float()
        z = time_stepping(x_support, alpha_support, u_init)
        u_rec = model.decoder(torch.from_numpy(x[:150]).float(), z).detach().numpy()

        images = []
        for i in range(i_test * Nt, (i_test + 1) * Nt):
            # figure without gui
            plt.ioff()
            # plot with gui

            fig = plt.figure(figsize=(8, 8 * 0.64))
            plt.plot(x[0], u[i], 'm', label='FOM')
            plt.plot(x[0], u_rec[i % Nt], 'c', label='Approximation')
            plt.plot(x[0, support_point_indices], u_rec[i % Nt, support_point_indices], 'co',
                     label='integration points')
            plt.axvspan(x[0, 0], x[0, 167], facecolor='grey',
                        alpha=alphas[i_test, 0] / alphas.max())
            plt.axvspan(x[0, 1 * 167], x[0, 2 * 167], facecolor='grey',
                        alpha=alphas[i_test, 1] / alphas.max())
            plt.axvspan(x[0, 2 * 167], x[0, -1], facecolor='grey',
                        alpha=alphas[i_test, 2] / alphas.max())
            plt.xlabel('$x$')
            plt.ylabel('$u$')
            plt.ylim([-1, 1])
            plt.xlim([0, x.max()])
            # legend with fixed position upper right
            plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
            plt.title(f'$t={t[i % Nt] * 1000:.3f}\,$ms')
            # increase font size
            plt.rcParams.update({'font.size': 16})
            # increase label and tick size
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tight_layout()
            fig.savefig('gifs/tmp.png', dpi=300)
            images.append(imageio.imread('gifs/tmp.png'))
        # repeated gif
        imageio.mimsave(f'gifs/diffusion_approx_{i_test + 1}.gif', images, loop=1)
