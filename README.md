# MNIST Image Embedding

This project aims to create a 2D embedding of the MNIST dataset using three different models: Denoise Autoencoder, Variational Autoencoder, and VQ-VAE. The goal is to compare the results of these models and explore their effectiveness in creating a compact representation of the MNIST images.

## Dataset

The dataset used in this project is MNIST, which is a widely used dataset in the field of machine learning. MNIST consists of a large number of grayscale images of handwritten digits from 0 to 9. Each image is 28x28 pixels in size.

## Methodology

The methodology involves training three different types of autoencoders on the MNIST dataset to create a 2D embedding of the images. The three models used are Denoise Autoencoder, Variational Autoencoder, and VQ-VAE.

Autoencoders are neural networks that are trained to reconstruct their input data. In this project, they are employed to learn a compressed representation of the MNIST images in a lower-dimensional space. By training the autoencoders to minimize the reconstruction error, we aim to obtain meaningful embeddings that capture the essential features of the images.

The Denoise Autoencoder is designed to handle noisy inputs by introducing noise during the training phase. It helps in learning robust representations by forcing the autoencoder to reconstruct the original images from the noisy versions.

The Variational Autoencoder (VAE) is a generative model that learns a probability distribution in the latent space. It introduces a regularizer to ensure that the learned distribution is close to a known prior distribution, typically a Gaussian distribution. The VAE provides a way to generate new samples by sampling from the learned distribution.

The Vector Quantized Variational Autoencoder (VQ-VAE) is an extension of the VAE that incorporates vector quantization. It uses a discrete codebook to quantize the continuous latent space, which allows for efficient encoding and decoding of the data. VQ-VAE has been shown to produce more structured and interpretable embeddings.

## Conclusion

In summary, this project showcases the utilization of Denoise Autoencoder, Variational Autoencoder, and VQ-VAE to construct a 2D representation of the MNIST dataset. By evaluating the outcomes of these models, we can acquire valuable understanding regarding their ability to capture the intrinsic characteristics of the images. The resulting embeddings offer diverse possibilities, such as visualization and clustering, and can be adapted to different data modalities like text and audio.