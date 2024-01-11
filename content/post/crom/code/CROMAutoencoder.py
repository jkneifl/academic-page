import torch
from torch import nn
import logging
# logging configuration
logging.basicConfig(level=logging.INFO, format='CROM: %(asctime)s %(message)s')


# create autoencoder
class CROMAutoencoder(nn.Module):
    def __init__(self, r, n_x, n_d=1, n_layers=1, n_neurons=128):
        """
        :param r: latent space dimension
        :param n_x: number of spatial points
        :param n_d: number of coordinates required to describe the position (1 for 1D, 2 for 2D, ...)
        :param n_layers: number of hidden layers in the encoder and decoder
        :param n_neurons: number of neurons in each hidden layer
        """
        super(CROMAutoencoder, self).__init__()
        self.r = r
        self.n_x = n_x
        self.n_d = n_d
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        # create layers for the encoder
        self.encoder_0 = nn.Linear(self.n_x, self.n_neurons)
        self.encoder_0_act = nn.ELU(True)
        # hidden layers
        for i in range(n_layers):
            setattr(self, f'encoder_{i + 1}', nn.Linear(self.n_neurons, self.n_neurons))
            setattr(self, f'encoder_{i + 1}_act', nn.ELU(True))
        setattr(self, f'encoder_{n_layers + 1}', nn.Linear(self.n_neurons, self.r))

        # create layers for decoder
        self.decoder_0 = nn.Linear(self.r + self.n_d, self.n_neurons)
        self.decoder_0_act = nn.ELU(True)
        # hidden layers
        for i in range(n_layers):
            setattr(self, f'decoder_{i + 1}', nn.Linear(self.n_neurons, self.n_neurons))
            setattr(self, f'decoder_{i + 1}_act', nn.ELU(True))
        # output layer with skip connection from spatial coordinate
        setattr(self, f'decoder_{n_layers + 1}', nn.Linear(self.n_neurons + self.n_d, 1))

    def encoder(self, u):
        """
        Encoder that maps the vector field values to the latent space
        :param u: (batchsize x n) vector field values
        :return:
        """
        for i in range(self.n_layers+2): # +2 for input and output layer
            u = getattr(self, f'encoder_{i}')(u)
            if hasattr(self, f'encoder_{i}_act'):
                u = getattr(self, f'encoder_{i}_act')(u)
        return u

    def decoder(self, x, z):
        """
        Decoder that maps the latent variable along with a spatial coordinate to the reconstructed vector field value
        :param x: (batchsize x n) positional coordinates
        :param z: (batchsize x r) latent variable
        :return:
        """

        # save batch size for reshaping later
        batch_size_local = x.size(0)

        # repeat z for every point in x (batch_size_local x n_x x r)
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        # add new axis to x to the end (batch_size_local x n_x x n_d)
        x = x.unsqueeze(2)
        # concatenate x and z
        decoder_input = torch.cat((x, z), dim=2)
        # reshape for decoder so that all the points are processed at once (batchsize = batch_size_local * n_x)
        decoder_input = decoder_input.view(-1, self.r + self.n_d)

        u_ = decoder_input
        for i in range(self.n_layers + 1): # +1 for input layer
            u_ = getattr(self, f'decoder_{i}')(u_)
            if hasattr(self, f'decoder_{i}_act'):
                u_ = getattr(self, f'decoder_{i}_act')(u_)
        # stack x and u_ (skip connection from x to output layer)
        output_input = torch.cat((x.view(-1, self.n_d), u_), dim=1)
        u_ = getattr(self, f'decoder_{self.n_layers + 1}')(output_input)

        # reshape x_rec
        u_rec = u_.view(batch_size_local, -1)

        return u_rec

    def forward(self, x, u):
        """
        :param x: (batchsize x n) positional coordinates
        :param u: (batchsize x n) vector field values
        """

        # encode input to latent space
        z = self.encoder(u)

        # reshape x_rec
        u_rec = self.decoder(x, z)

        return u_rec

    def fit(self, train_loader, num_epochs, criterion=nn.MSELoss(), optimizer=None):
        """
        fit the crom autoencoder
        :param train_loader: torch dataloader
        :param num_epochs: number of epochs
        :param criterion: torch loss function
        :param optimizer: torch optimizer
        :return:
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        losses = []
        for epoch in range(num_epochs):
            for data in train_loader:
                x_, u_ = data
                # forward
                output = self(x_, u_)
                loss = criterion(output, u_)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # log
            logging.info('epoch [{}/{}], loss:{:.6f}'
                  .format(epoch + 1, num_epochs, loss.item()))
            losses.append(loss.item())

        # save model
        torch.save(self.state_dict(), f'model.pth')

        return losses
