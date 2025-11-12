#!/usr/bin/env python3
"""
Arquiteturas de GANs suportadas
"""

import torch
import torch.nn as nn

# ====================================================================================
# DCGAN (Deep Convolutional GAN)
# ====================================================================================


class DCGANGenerator(nn.Module):
    """Gerador DCGAN - gera imagens a partir de ruído"""

    def __init__(self, nz=100, ngf=64, nc=3, img_size=64):
        super(DCGANGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.img_size = img_size

        num_layers = int(torch.log2(torch.tensor(img_size))) - 2

        layers = []
        current_dim = ngf * (2 ** (num_layers - 1))

        layers.extend(
            [
                nn.ConvTranspose2d(nz, current_dim, 4, 1, 0, bias=False),
                nn.BatchNorm2d(current_dim),
                nn.ReLU(True),
            ]
        )

        for i in range(num_layers - 1):
            next_dim = current_dim // 2
            layers.extend(
                [
                    nn.ConvTranspose2d(current_dim, next_dim, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_dim),
                    nn.ReLU(True),
                ]
            )
            current_dim = next_dim

        layers.extend(
            [
                nn.ConvTranspose2d(current_dim, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        )

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class DCGANDiscriminator(nn.Module):
    """Discriminador DCGAN - classifica imagens como real ou fake"""

    def __init__(self, ndf=64, nc=3, img_size=64):
        super(DCGANDiscriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.img_size = img_size

        num_layers = int(torch.log2(torch.tensor(img_size))) - 3

        layers = []

        layers.extend(
            [
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        current_dim = ndf
        for i in range(num_layers):
            next_dim = current_dim * 2
            layers.extend(
                [
                    nn.Conv2d(current_dim, next_dim, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_dim = next_dim

        layers.extend(
            [
                nn.Conv2d(current_dim, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            ]
        )

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


# ====================================================================================
# DCGAN CONDICIONAL (label → imagem)
# ====================================================================================


class ConditionalDCGANGenerator(nn.Module):
    """
    DCGAN condicional simples.
    Usa embedding de classe para condicionar o ruído.
    """

    def __init__(self, nz=100, ngf=64, nc=3, img_size=64, num_classes=10):
        super().__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, nz)

        num_layers = int(torch.log2(torch.tensor(img_size))) - 2
        layers = []
        current_dim = ngf * (2 ** (num_layers - 1))

        # Entrada continua sendo (nz x 1 x 1), mas já "contaminado" pelo label
        layers.extend(
            [
                nn.ConvTranspose2d(nz, current_dim, 4, 1, 0, bias=False),
                nn.BatchNorm2d(current_dim),
                nn.ReLU(True),
            ]
        )

        for i in range(num_layers - 1):
            next_dim = current_dim // 2
            layers.extend(
                [
                    nn.ConvTranspose2d(current_dim, next_dim, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_dim),
                    nn.ReLU(True),
                ]
            )
            current_dim = next_dim

        layers.extend(
            [
                nn.ConvTranspose2d(current_dim, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        )

        self.main = nn.Sequential(*layers)

    def forward(self, noise, labels):
        # labels: (B,)
        emb = self.label_emb(labels).unsqueeze(2).unsqueeze(3)  # (B, nz, 1, 1)
        z = noise + emb
        return self.main(z)


class ConditionalDCGANDiscriminator(nn.Module):
    """
    Discriminador condicional:
    concatena um mapa derivado do label como canal extra.
    """

    def __init__(self, ndf=64, nc=3, img_size=64, num_classes=10):
        super().__init__()
        self.ndf = ndf
        self.nc = nc
        self.img_size = img_size
        self.num_classes = num_classes

        # Cada classe vira um mapa (H*W) que será reshape para (1,H,W)
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        num_layers = int(torch.log2(torch.tensor(img_size))) - 3
        in_channels = nc + 1  # imagem + canal de condição

        layers = []
        layers.extend(
            [
                nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        current_dim = ndf
        for i in range(num_layers):
            next_dim = current_dim * 2
            layers.extend(
                [
                    nn.Conv2d(current_dim, next_dim, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_dim = next_dim

        layers.extend(
            [
                nn.Conv2d(current_dim, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            ]
        )

        self.main = nn.Sequential(*layers)

    def forward(self, img, labels):
        b = img.size(0)
        cond = self.label_emb(labels).view(b, 1, self.img_size, self.img_size)
        x = torch.cat([img, cond], dim=1)
        return self.main(x).view(-1, 1).squeeze(1)


# ====================================================================================
# WGAN-GP
# ====================================================================================


class WGANGenerator(nn.Module):
    """Gerador para WGAN-GP"""

    def __init__(self, nz=100, ngf=64, nc=3, img_size=64):
        super(WGANGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.img_size = img_size

        num_layers = int(torch.log2(torch.tensor(img_size))) - 2

        layers = []
        current_dim = ngf * (2 ** (num_layers - 1))

        layers.extend(
            [
                nn.ConvTranspose2d(nz, current_dim, 4, 1, 0, bias=False),
                nn.BatchNorm2d(current_dim),
                nn.ReLU(True),
            ]
        )

        for i in range(num_layers - 1):
            next_dim = current_dim // 2
            layers.extend(
                [
                    nn.ConvTranspose2d(current_dim, next_dim, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_dim),
                    nn.ReLU(True),
                ]
            )
            current_dim = next_dim

        layers.extend(
            [
                nn.ConvTranspose2d(current_dim, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        )

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class WGANCritic(nn.Module):
    """Crítico para WGAN-GP"""

    def __init__(self, ndf=64, nc=3, img_size=64):
        super(WGANCritic, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.img_size = img_size

        num_layers = int(torch.log2(torch.tensor(img_size))) - 3

        layers = []

        layers.extend(
            [
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        current_dim = ndf
        for i in range(num_layers):
            next_dim = current_dim * 2
            layers.extend(
                [
                    nn.Conv2d(current_dim, next_dim, 4, 2, 1, bias=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_dim = next_dim

        layers.append(nn.Conv2d(current_dim, 1, 4, 1, 0, bias=True))

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input).view(-1)


# ====================================================================================
# Helpers
# ====================================================================================


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_model(model_type, model_config):
    """
    Factory: 'dcgan', 'wgan-gp', 'dcgan-cond'
    """
    nz = model_config.get("nz", 100)
    ngf = model_config.get("ngf", 64)
    ndf = model_config.get("ndf", 64)
    nc = model_config.get("nc", 3)
    img_size = model_config.get("img_size", 64)

    mt = model_type.lower()

    if mt == "dcgan":
        generator = DCGANGenerator(nz=nz, ngf=ngf, nc=nc, img_size=img_size)
        discriminator = DCGANDiscriminator(ndf=ndf, nc=nc, img_size=img_size)
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        return generator, discriminator

    elif mt in ("dcgan-cond", "dcgan_cond", "cgan"):
        num_classes = model_config.get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes é obrigatório para dcgan-cond")
        generator = ConditionalDCGANGenerator(
            nz=nz, ngf=ngf, nc=nc, img_size=img_size, num_classes=num_classes
        )
        discriminator = ConditionalDCGANDiscriminator(
            ndf=ndf, nc=nc, img_size=img_size, num_classes=num_classes
        )
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        return generator, discriminator

    elif mt == "wgan-gp":
        generator = WGANGenerator(nz=nz, ngf=ngf, nc=nc, img_size=img_size)
        critic = WGANCritic(ndf=ndf, nc=nc, img_size=img_size)
        generator.apply(weights_init)
        critic.apply(weights_init)
        return generator, critic

    else:
        raise ValueError(
            f"Modelo '{model_type}' não suportado. Use 'dcgan', 'dcgan-cond' ou 'wgan-gp'"
        )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
