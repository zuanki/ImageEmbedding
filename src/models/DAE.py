import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(DenoisingAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(8, 2, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(2, 1, kernel_size=2, stride=1, padding=0),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(4, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.PReLU(),
            nn.Linear(4, 1*7*7),
            nn.PReLU(),
            nn.Unflatten(1, (1, 7, 7)),
            nn.ConvTranspose2d(1, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def __str__(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super().__str__() + f'\nTrainable parameters: {n_params}'
