import torch
from _types import *
from torch import nn
from encoders import BaseVAE
from torch.nn import functional


class VanillaVAE(BaseVAE):
    def __init__(self,
                 latent_dimensions: int) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dimensions = latent_dimensions

        # Encoder
        modules = []
        hidden_dimensions = [32, 64, 128, 256, 512]
        in_channel = 1
        for dim in hidden_dimensions:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channel, out_channels=dim,
                        kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm3d(dim),
                    nn.LeakyReLU(),
                )
            )

            in_channel = dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(
            hidden_dimensions[-1] * 64, latent_dimensions)
        self.fc_var = nn.Linear(
            hidden_dimensions[-1] * 64, latent_dimensions)

        # Decoder
        modules = []

        self.decoder_input = nn.Linear(
            latent_dimensions, hidden_dimensions[-1] * 64)
        hidden_dimensions.reverse()
        in_channel = hidden_dimensions[0]
        hidden_dimensions = hidden_dimensions[1:]
        for dim in hidden_dimensions:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_channel,
                        out_channels=dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm3d(dim),
                    nn.LeakyReLU()
                )
            )

            in_channel = dim

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(
                hidden_dimensions[-1],
                hidden_dimensions[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm3d(hidden_dimensions[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(
                hidden_dimensions[-1],
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
            nn.Tanh(),
        )

    def encode(self, input_tensor: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input_tensor: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input_tensor)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 4, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Re-parameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.

        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = functional.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': -kld_loss.detach()
        }

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]
    