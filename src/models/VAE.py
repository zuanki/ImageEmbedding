import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        # input shape: [batch_size, 28, 28]

        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder_fc1 = nn.Linear(28 * 28, 256)
        self.encoder_fc2_mean = nn.Linear(256, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(256, latent_dim)

        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim, 256)
        self.decoder_fc2 = nn.Linear(256, 28 * 28)

    def encode(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.encoder_fc1(x))

        mean = self.encoder_fc2_mean(x)
        logvar = self.encoder_fc2_logvar(x)

        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mean + eps * std

        return z

    def decode(self, z):
        x = F.relu(self.decoder_fc1(z))
        x = torch.sigmoid(self.decoder_fc2(x))

        x = x.view(-1, 1, 28, 28)

        return x

    def forward(self, x):
        mean, logvar = self.encode(x)

        z = self.reparameterize(mean, logvar)

        recon_x = self.decode(z)

        return recon_x, mean, logvar
