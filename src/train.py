#!/usr/bin/env python3
"""
Script aprimorado para treinar DCGAN com diferentes datasets
"""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3, img_size=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # Calcular número de camadas baseado no tamanho da imagem
        num_layers = int(torch.log2(torch.tensor(img_size))) - 2

        layers = []
        # Primeira camada
        layers.extend(
            [
                nn.ConvTranspose2d(nz, ngf * (2**num_layers), 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * (2**num_layers)),
                nn.ReLU(True),
            ]
        )

        # Camadas intermediárias
        for i in range(num_layers):
            power = num_layers - i
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        ngf * (2**power), ngf * (2 ** (power - 1)), 4, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(ngf * (2 ** (power - 1))),
                    nn.ReLU(True),
                ]
            )

        # Última camada
        layers.extend([nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh()])

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf=64, nc=3, img_size=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        num_layers = int(torch.log2(torch.tensor(img_size))) - 2

        layers = []
        # Primeira camada
        layers.extend(
            [nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
        )

        # Camadas intermediárias
        for i in range(num_layers):
            layers.extend(
                [
                    nn.Conv2d(ndf * (2**i), ndf * (2 ** (i + 1)), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * (2 ** (i + 1))),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        # Última camada
        layers.append(nn.Conv2d(ndf * (2**num_layers), 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


def get_dataset(name, dataroot, image_size):
    """Configuração melhorada dos datasets"""
    if name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = dset.CIFAR10(root=dataroot, download=True, transform=transform)
        nc = 3

    elif name == "fashion":
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        dataset = dset.FashionMNIST(root=dataroot, download=True, transform=transform)
        nc = 1

    elif name == "anime":
        # Assumindo que as imagens estão em data/anime/
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = dset.ImageFolder(
            root=os.path.join(dataroot, "anime"), transform=transform
        )
        nc = 3

    else:
        raise ValueError(f"Dataset '{name}' não suportado")

    return dataset, nc


def save_config(args, nc, output_dir):
    """Salva configuração do modelo"""
    config = {
        "dataset": args.dataset,
        "image_size": args.image_size,
        "nc": nc,
        "nz": args.nz,
        "ngf": args.ngf,
        "ndf": args.ndf,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta1": args.beta1,
        "epochs": args.epochs,
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "fashion", "anime"],
        help="Dataset para treinamento",
    )
    parser.add_argument(
        "--dataroot", default="./data", help="Diretório raiz dos datasets"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Número de workers para carregamento de dados",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Tamanho do batch")
    parser.add_argument(
        "--image-size", type=int, default=64, help="Tamanho das imagens (quadradas)"
    )
    parser.add_argument(
        "--nz", type=int, default=100, help="Tamanho do vetor de ruído latente"
    )
    parser.add_argument(
        "--ngf", type=int, default=64, help="Número de filtros do gerador"
    )
    parser.add_argument(
        "--ndf", type=int, default=64, help="Número de filtros do discriminador"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Número de épocas de treinamento"
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="Taxa de aprendizado")
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Beta1 para otimizador Adam"
    )
    parser.add_argument("--ngpu", type=int, default=1, help="Número de GPUs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Diretório para salvar resultados",
    )

    args = parser.parse_args()

    # Criar diretórios
    model_dir = os.path.join("models", args.dataset)
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Configurar device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.ngpu > 0 else "cpu"
    )
    print(f"Device: {device}")

    # Carregar dataset
    dataset, nc = get_dataset(args.dataset, args.dataroot, args.image_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    # Criar modelos
    netG = Generator(args.ngpu, args.nz, args.ngf, nc, args.image_size).to(device)
    netD = Discriminator(args.ngpu, args.ndf, nc, args.image_size).to(device)

    # Critério e otimizadores
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Treinar
    print(f"Iniciando treinamento do {args.dataset}...")
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # Treinar Discriminador
            ###########################
            netD.zero_grad()
            real = data[0].to(device)
            batch_size = real.size(0)
            label_real = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            label_fake = torch.full((batch_size,), 0, dtype=torch.float, device=device)

            output = netD(real)
            errD_real = criterion(output, label_real)
            errD_real.backward()

            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, label_fake)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # Treinar Gerador
            ###########################
            netG.zero_grad()
            output = netD(fake)
            errG = criterion(output, label_real)
            errG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(
                    f"[{epoch}/{args.epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}"
                )

        # Salvar imagens a cada época
        with torch.no_grad():
            fake = netG(fixed_noise)
            vutils.save_image(
                fake.detach(),
                f"{output_dir}/fake_samples_epoch_{epoch:03d}.png",
                normalize=True,
            )

        # Salvar modelo a cada 10 épocas e na última
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save(netG.state_dict(), f"{model_dir}/netG_epoch_{epoch:03d}.pth")
            torch.save(netD.state_dict(), f"{model_dir}/netD_epoch_{epoch:03d}.pth")

    # Salvar configuração
    save_config(args, nc, model_dir)
    print("Treinamento concluído!")


if __name__ == "__main__":
    main()
