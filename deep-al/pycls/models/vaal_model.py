# This file is modified  taken from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init as init

import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, device_id,z_dim=32, nc=3):
        super(VAE, self).__init__()
        print("============================")
        logger.info("============================")
        print(f"Constructing VAE MODEL with z_dim: {z_dim}")
        logger.info(f"Constructing VAE MODEL with z_dim: {z_dim}")
        print("============================")
        logger.info("============================")
        self.encode_shape = int(z_dim/16)
        if z_dim == 32:
            self.decode_shape = 4
        elif z_dim == 64:
            self.decode_shape = 8
        else:
            self.decode_shape = 4
        self.device_id = device_id
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32 or B, 128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16 or B, 256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8 or B, 512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4 or B, 1024, 8, 8
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*self.encode_shape*self.encode_shape)),                                 # B, 1024*4*4 or B, 1024, 4, 4
        )

        self.fc_mu = nn.Linear(1024*self.encode_shape*self.encode_shape, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*self.encode_shape*self.encode_shape, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024*self.decode_shape*self.decode_shape),                           # B, 1024*8*8
            View((-1, 1024, self.decode_shape, self.decode_shape)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        #if mu.is_cuda:
        stds, epsilon = stds.cuda(), epsilon.cuda()
        mu = mu.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class clf_Discriminator(nn.Module):
    """
    Model to circumvent the need of learning a separate task learner by 
    combining the discriminator and task classifier.
    """
    def __init__(self, z_dim=10, n_classes=10):
        super(clf_Discriminator, self).__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True)
        )

        self.disc_out = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.clf_out = nn.Sequential(
            nn.Linear(512, self.n_classes),
        )
        
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        z = self.net(z)
        disc_out = self.disc_out(z)
        clf_out = self.clf_out(z)
        return disc_out, clf_out


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.penultimate_active = False
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True)
        )

        self.out = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        z = self.net(z)
        if self.penultimate_active:
            return z, self.out(z)
        return self.out(z)

class WGAN_Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(WGAN_Discriminator, self).__init__()
        self.z_dim = z_dim
        self.penultimate_active = False
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True)
        )

        self.out = nn.Sequential(
            nn.Linear(512, 1),
            #nn.Sigmoid()
        )
        
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        z = self.net(z)
        if self.penultimate_active:
            return z, self.out(z)
        return self.out(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
