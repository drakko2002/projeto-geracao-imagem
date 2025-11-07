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

        # Calcular número de camadas baseado no tamanho da imagem
        # img_size = 64: log2(64)=6, então 6-2=4 camadas totais de upsampling
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 (4 upsampling layers)
        num_layers = int(torch.log2(torch.tensor(img_size))) - 2

        layers = []

        # Camada inicial: transforma vetor z (nz x 1 x 1) em feature map 4x4
        current_dim = ngf * (2 ** (num_layers - 1))
        layers.extend(
            [
                nn.ConvTranspose2d(nz, current_dim, 4, 1, 0, bias=False),
                nn.BatchNorm2d(current_dim),
                nn.ReLU(True),
            ]
        )

        # Camadas de upsampling intermediárias (num_layers - 1 camadas)
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

        # Camada final: última upsampling + mudança para nc canais
        layers.extend(
            [nn.ConvTranspose2d(current_dim, nc, 4, 2, 1, bias=False), nn.Tanh()]
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

        # Calcular número de camadas (deve ser simétrico ao gerador)
        # Para img_size=64: log2(64)=6, então 6-3=3 camadas intermediárias
        num_layers = int(torch.log2(torch.tensor(img_size))) - 3

        layers = []

        # Camada inicial: 64x64 -> 32x32
        layers.extend(
            [nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
        )

        # Camadas intermediárias de downsampling
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

        # Camada final: reduz para 1x1 e classifica como real (1) ou fake (0)
        layers.extend([nn.Conv2d(current_dim, 1, 4, 1, 0, bias=False), nn.Sigmoid()])

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


# ====================================================================================
# WGAN-GP (Wasserstein GAN with Gradient Penalty)
# ====================================================================================


class WGANGenerator(nn.Module):
    """Gerador para WGAN-GP"""

    def __init__(self, nz=100, ngf=64, nc=3, img_size=64):
        super(WGANGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.img_size = img_size

        # Mesmo cálculo que DCGAN
        num_layers = int(torch.log2(torch.tensor(img_size))) - 2

        layers = []
        current_dim = ngf * (2 ** (num_layers - 1))

        # Camada inicial
        layers.extend(
            [
                nn.ConvTranspose2d(nz, current_dim, 4, 1, 0, bias=False),
                nn.BatchNorm2d(current_dim),
                nn.ReLU(True),
            ]
        )

        # Camadas intermediárias
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

        # Camada final
        layers.extend(
            [nn.ConvTranspose2d(current_dim, nc, 4, 2, 1, bias=False), nn.Tanh()]
        )

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class WGANCritic(nn.Module):
    """Crítico para WGAN-GP (sem sigmoid, retorna score real)"""

    def __init__(self, ndf=64, nc=3, img_size=64):
        super(WGANCritic, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.img_size = img_size

        # Mesmo cálculo que o discriminador DCGAN
        num_layers = int(torch.log2(torch.tensor(img_size))) - 3

        layers = []

        # Camada inicial (sem BatchNorm para WGAN-GP)
        layers.extend(
            [nn.Conv2d(nc, ndf, 4, 2, 1, bias=True), nn.LeakyReLU(0.2, inplace=True)]
        )

        # Camadas intermediárias (sem BatchNorm/LayerNorm para estabilidade WGAN-GP)
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

        # Camada final (sem sigmoid - retorna score Wasserstein)
        layers.append(nn.Conv2d(current_dim, 1, 4, 1, 0, bias=True))

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input).view(-1)


# ====================================================================================
# Funções auxiliares
# ====================================================================================


def weights_init(m):
    """Inicialização de pesos conforme paper DCGAN"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_model(model_type, model_config):
    """
    Factory function para criar modelos

    Args:
        model_type: 'dcgan' ou 'wgan-gp'
        model_config: dicionário com configurações do modelo

    Returns:
        (generator, discriminator/critic)
    """
    nz = model_config.get("nz", 100)
    ngf = model_config.get("ngf", 64)
    ndf = model_config.get("ndf", 64)
    nc = model_config.get("nc", 3)
    img_size = model_config.get("img_size", 64)

    if model_type.lower() == "dcgan":
        generator = DCGANGenerator(nz=nz, ngf=ngf, nc=nc, img_size=img_size)
        discriminator = DCGANDiscriminator(ndf=ndf, nc=nc, img_size=img_size)

        # Inicializar pesos
        generator.apply(weights_init)
        discriminator.apply(weights_init)

        return generator, discriminator

    elif model_type.lower() == "wgan-gp":
        generator = WGANGenerator(nz=nz, ngf=ngf, nc=nc, img_size=img_size)
        critic = WGANCritic(ndf=ndf, nc=nc, img_size=img_size)

        # Inicializar pesos
        generator.apply(weights_init)
        critic.apply(weights_init)

        return generator, critic

    else:
        raise ValueError(
            f"Modelo '{model_type}' não suportado. Use 'dcgan' ou 'wgan-gp'"
        )


def count_parameters(model):
    """Conta número de parâmetros treináveis no modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
