#!/usr/bin/env python3
"""
Configurações e datasets para treinamento
"""

import os

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ====================================================================================
# Configurações de Datasets
# ====================================================================================

DATASET_CONFIGS = {
    "cifar10": {
        "name": "CIFAR-10",
        "description": "Imagens coloridas 32x32 de 10 categorias",
        "classes": [
            "Aviões",
            "Carros",
            "Pássaros",
            "Gatos",
            "Cervos",
            "Cachorros",
            "Sapos",
            "Cavalos",
            "Navios",
            "Caminhões",
        ],
        "nc": 3,  # RGB
        "default_img_size": 128,  # Padrão 128px, suporta preset 256px
        "download": True,
    },
    "mnist": {
        "name": "MNIST",
        "description": "Dígitos escritos à mão 28x28 em escala de cinza",
        "classes": ["0","1","2","3","4","5","6","7","8","9"],
        "nc": 1,  # Grayscale
        "default_img_size": 128,  # Padrão 128px, suporta preset 256px
        "download": True,
    },
    "fashion-mnist": {
        "name": "Fashion-MNIST",
        "description": "Imagens 28x28 de roupas e acessórios em escala de cinza",
        "classes": [
            "Camiseta",
            "Calça",
            "Suéter",
            "Vestido",
            "Casaco",
            "Sandália",
            "Camisa",
            "Tênis",
            "Bolsa",
            "Bota",
        ],
        "nc": 1,  # Grayscale
        "default_img_size": 128,  # Padrão 128px, suporta preset 256px
        "download": True,
    },
    "celeba": {
        "name": "CelebA",
        "description": "Imagens de celebridades (requer download manual)",
        "classes": ["Faces de celebridades"],
        "nc": 3,  # RGB
        "default_img_size": 128,  # Padrão 128px, suporta preset 256px
        "download": False,  # Requer download manual
    },
    "custom": {
        "name": "Custom Dataset",
        "description": "Dataset customizado (pasta de imagens)",
        "classes": ["Imagens customizadas"],
        "nc": 3,  # RGB (pode ser ajustado)
        "default_img_size": 128,  # Padrão 128px, suporta preset 256px
        "download": False,
    },
}


# ====================================================================================
# Configurações de Modelos
# ====================================================================================

MODEL_CONFIGS = {
    "dcgan": {
        "name": "DCGAN",
        "description": "Deep Convolutional GAN (Radford et al., 2015)",
        "default_lr": 0.0002,
        "default_beta1": 0.5,
        "default_beta2": 0.999,
        "default_nz": 100,
        "default_ngf": 64,
        "default_ndf": 64,
    },
    "wgan-gp": {
        "name": "WGAN-GP",
        "description": "Wasserstein GAN with Gradient Penalty (Gulrajani et al., 2017)",
        "default_lr": 0.0001,
        "default_beta1": 0.0,
        "default_beta2": 0.9,
        "default_nz": 100,
        "default_ngf": 64,
        "default_ndf": 64,
        "n_critic": 5,  # Treinar critic N vezes por iteração do gerador
        "lambda_gp": 10.0,  # Peso do gradient penalty
    },
}


# ====================================================================================
# Funções para criar datasets
# ====================================================================================


def get_dataset(
    dataset_name, dataroot="./data", img_size=64, batch_size=128, workers=2
):
    """
    Cria e retorna dataset e dataloader

    Args:
        dataset_name: nome do dataset ('cifar10', 'mnist', etc.)
        dataroot: diretório raiz para salvar/carregar datasets
        img_size: tamanho das imagens (redimensionadas para img_size x img_size)
        batch_size: tamanho do batch
        workers: número de workers para DataLoader

    Returns:
        (dataloader, nc) - dataloader e número de canais
    """

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Dataset '{dataset_name}' não suportado. "
            f"Datasets disponíveis: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    nc = config["nc"]

    # Criar diretório se não existir
    os.makedirs(dataroot, exist_ok=True)

    # Transformações baseadas no número de canais
    if nc == 1:  # Grayscale
        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    else:  # RGB
        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    # Criar dataset específico
    if dataset_name == "cifar10":
        dataset = dset.CIFAR10(
            root=dataroot, download=config["download"], transform=transform
        )

    elif dataset_name == "mnist":
        dataset = dset.MNIST(
            root=dataroot, download=config["download"], transform=transform
        )

    elif dataset_name == "fashion-mnist":
        dataset = dset.FashionMNIST(
            root=dataroot, download=config["download"], transform=transform
        )

    elif dataset_name == "celeba":
        # CelebA requer download manual
        celeba_path = os.path.join(dataroot, "celeba")
        if not os.path.exists(celeba_path):
            raise ValueError(
                f"CelebA dataset não encontrado em {celeba_path}.\n"
                "Por favor, baixe o dataset de http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
            )
        dataset = dset.ImageFolder(root=celeba_path, transform=transform)

    elif dataset_name == "custom":
        # Dataset customizado (pasta de imagens)
        custom_path = os.path.join(dataroot, "custom")
        if not os.path.exists(custom_path):
            raise ValueError(
                f"Dataset customizado não encontrado em {custom_path}.\n"
                "Por favor, crie a pasta e adicione suas imagens em subpastas."
            )
        dataset = dset.ImageFolder(root=custom_path, transform=transform)

    else:
        raise ValueError(f"Dataset '{dataset_name}' não implementado")

    # Criar DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True,  # Importante para manter batch_size consistente
        pin_memory=True,  # Acelera transferência CPU -> GPU
    )

    return dataloader, nc


def get_model_config(model_type):
    """Retorna configuração padrão para um tipo de modelo"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Modelo '{model_type}' não suportado. "
            f"Modelos disponíveis: {list(MODEL_CONFIGS.keys())}"
        )

    return MODEL_CONFIGS[model_type].copy()


def get_dataset_info(dataset_name):
    """Retorna informações sobre um dataset"""
    if dataset_name not in DATASET_CONFIGS:
        return None
    return DATASET_CONFIGS[dataset_name].copy()


def list_available_datasets():
    """Lista todos os datasets disponíveis"""
    print("\n" + "=" * 70)
    print("DATASETS DISPONÍVEIS")
    print("=" * 70)

    for name, config in DATASET_CONFIGS.items():
        print(f"\n📦 {name}")
        print(f"   Nome: {config['name']}")
        print(f"   Descrição: {config['description']}")
        print(
            f"   Canais: {config['nc']} ({'RGB' if config['nc'] == 3 else 'Grayscale'})"
        )
        print(f"   Download automático: {'Sim' if config['download'] else 'Não'}")
        if config["classes"]:
            print(
                f"   Classes: {', '.join(config['classes'][:3])}{'...' if len(config['classes']) > 3 else ''}"
            )

    print("\n" + "=" * 70)


def list_available_models():
    """Lista todos os modelos disponíveis"""
    print("\n" + "=" * 70)
    print("MODELOS DISPONÍVEIS")
    print("=" * 70)

    for name, config in MODEL_CONFIGS.items():
        print(f"\n🤖 {name}")
        print(f"   Nome: {config['name']}")
        print(f"   Descrição: {config['description']}")
        print(f"   Learning rate padrão: {config['default_lr']}")
        print(f"   Beta1 padrão: {config['default_beta1']}")

    print("\n" + "=" * 70)
