#!/usr/bin/env python3
import torch
import torch.nn as nn

class vannillaVariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim = 200, z_dim = 20):
        super().__init__()
        self.tanh = nn.Tanh()

        # encoder
        self.img_2_hid = nn.Linear(input_dim,h_dim)
        self.hid_2_infer_mu = nn.Linear(h_dim,z_dim)
        self.hid_2_infer_sigma = nn.Linear(h_dim,z_dim)

        # decoder
        self.z_2_hid = nn.Linear(z_dim, h_dim)
        self.hid_2_gene_x = nn.Linear(h_dim, input_dim)

    def encoder(self, x):
        # q_phi(z|x)
        h = self.tanh(self.img_2_hid(x))
        mu = self.hid_2_infer_mu(h)
        sigma = self.hid_2_infer_sigma(h)
        return mu, sigma

    def decoder(self, z):
        # p_theta(x|z)
        h = self.tanh(self.z_2_hid(z))
        return torch.sigmoid(self.hid_2_gene_x(h))

    def forward(self, x):
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        img = self.decoder(z)
        return img, mu, sigma

if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)
    vae = vannillaVariationalAutoEncoder(input_dim=784)
    img, mu, sigma = vae(x)
    print(img.shape)
    print(mu.shape)
    print(sigma.shape)
