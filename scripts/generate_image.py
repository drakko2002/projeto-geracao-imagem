#!/usr/bin/env python3
"""
Script para gerar imagens únicas usando modelos DCGAN treinados
"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torchvision.utils as vutils


class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


def get_dataset_info(dataset_name):
    """Retorna informações específicas do dataset"""
    dataset_info = {
        "cifar10": {
            "nc": 3,  # Número de canais (RGB)
            "name": "CIFAR-10",
            "description": "Imagem colorida de objetos/veículos",
        },
        "mnist": {
            "nc": 1,  # Número de canais (escala de cinza)
            "name": "MNIST",
            "description": "Dígito escrito à mão",
        },
        "fashion-mnist": {
            "nc": 1,  # Número de canais (escala de cinza)
            "name": "Fashion-MNIST",
            "description": "Peça de roupa ou acessório",
        },
    }
    return dataset_info.get(dataset_name)


def generate_image(model_path, output_path, seed=None):
    """Gera uma única imagem usando o modelo especificado"""

    # Identificar o dataset baseado no caminho do modelo
    dataset_name = model_path.split("/")[-2]  # Pega o nome do dataset do caminho
    dataset_info = get_dataset_info(dataset_name)

    if dataset_info is None:
        print(f"❌ Dataset não reconhecido no caminho: {dataset_name}")
        return False

    # Configurar seed se fornecida
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        print(f"🎲 Usando seed: {seed}")

    # Configurar dispositivo (GPU/CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Usando dispositivo: {device}")

    # Criar e carregar o modelo
    print(f"📦 Carregando modelo: {os.path.basename(model_path)}")
    netG = Generator(ngpu=1, nc=dataset_info["nc"]).to(device)

    try:
        netG.load_state_dict(torch.load(model_path))
        print("✅ Modelo carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

    netG.eval()  # Modo de avaliação

    # Gerar ruído aleatório
    print("🎨 Gerando imagem...")
    noise = torch.randn(1, 100, 1, 1, device=device)

    # Gerar a imagem
    with torch.no_grad():
        fake = netG(noise)
        # Normalizar e salvar a imagem
        vutils.save_image(fake[0], output_path, normalize=True)

    print(f"✨ Nova {dataset_info['description']} gerada!")
    print(f"💾 Imagem salva em: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Gerador de imagens com DCGAN")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "mnist", "fashion-mnist"],
        help="Dataset usado no treinamento",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="Época específica para usar (-1 para última época disponível)",
    )
    parser.add_argument("--seed", type=int, help="Seed para geração (opcional)")
    parser.add_argument(
        "--output", type=str, help="Nome do arquivo de saída (opcional)"
    )

    args = parser.parse_args()

    # Diretório base dos modelos
    base_dir = f"outputs/{args.dataset}"

    # Encontrar modelos disponíveis
    if not os.path.exists(base_dir):
        print(f"❌ Nenhum modelo encontrado para {args.dataset}")
        return

    # Listar todos os modelos do gerador
    models = [
        f
        for f in os.listdir(base_dir)
        if f.startswith("netG_epoch_") and f.endswith(".pth")
    ]
    if not models:
        print(f"❌ Nenhum modelo do gerador encontrado em {base_dir}")
        return

    # Ordenar modelos por época
    models.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Selecionar modelo
    if args.epoch == -1:
        # Usar último modelo
        model_file = models[-1]
    else:
        # Procurar modelo da época específica
        model_file = f"netG_epoch_{args.epoch}.pth"
        if model_file not in models:
            print(f"❌ Modelo da época {args.epoch} não encontrado")
            print(
                f"📋 Épocas disponíveis: {[int(m.split('_')[-1].split('.')[0]) for m in models]}"
            )
            return

    # Definir nome do arquivo de saída
    if args.output:
        output_file = args.output
    else:
        epoch_num = int(model_file.split("_")[-1].split(".")[0])
        output_file = f"gerado_{args.dataset}_epoca{epoch_num}.png"

    # Gerar imagem
    model_path = os.path.join(base_dir, model_file)
    generate_image(model_path, output_file, args.seed)


if __name__ == "__main__":
    main()
