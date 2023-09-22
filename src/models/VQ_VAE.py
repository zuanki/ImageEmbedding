import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        # Flatten the input tensor
        inputs_flat = inputs.view(-1, self.embedding_dim)

        # Compute L2 distances between inputs and embedding vectors
        distances = torch.sum(inputs_flat ** 2, dim=1, keepdim=True) - 2 * torch.matmul(
            inputs_flat, self.embedding.weight.t()) + torch.sum(self.embedding.weight ** 2, dim=1)
        indices = torch.argmin(distances, dim=1)

        # Quantize the inputs based on the nearest embedding vectors
        quantized = self.embedding(indices).view(inputs.size())

        return quantized, indices


class ResidualLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs):
        x = torch.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x + inputs


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_fc1 = nn.Linear(28, 16)
        self.encoder_fc2_mean = nn.Linear(16, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(16, latent_dim)

    def forward(self, inputs):
        x = torch.relu(self.encoder_fc1(inputs))
        mean = self.encoder_fc2_mean(x)
        logvar = self.encoder_fc2_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder_fc1 = nn.Linear(latent_dim, 16)
        self.decoder_fc2 = nn.Linear(16, 28)

    def forward(self, inputs):
        x = torch.relu(self.decoder_fc1(inputs))
        x = torch.sigmoid(self.decoder_fc2(x))
        return x


class VQVAE2(nn.Module):
    def __init__(self, latent_dim):
        super(VQVAE2, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=10, embedding_dim=latent_dim)
        self.residual_layer = ResidualLayer(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, inputs):
        # Encoder
        enc_mean, enc_logvar = self.encoder(inputs)

        # Vector quantization
        quantized, indices = self.vector_quantizer(enc_mean)

        # Residual layer
        quantized_res = self.residual_layer(quantized)

        # Decoder
        dec_outputs = self.decoder(quantized_res)

        return dec_outputs, enc_mean, enc_logvar, quantized, indices
