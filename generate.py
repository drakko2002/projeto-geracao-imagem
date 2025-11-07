#!/usr/bin/env python3
"""
Script para gerar imagens usando modelos treinados

Uso:
    python generate.py --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth --num-samples 64
    python generate.py --checkpoint outputs/mnist/wgan-gp_xxx/checkpoints/checkpoint_epoch_50.pth --num-samples 100
"""

import argparse
import json
import os

import torch

from models import get_model
from utils import generate_samples


def main():
    parser = argparse.ArgumentParser(description="Gerar imagens usando modelo treinado")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Caminho para o checkpoint do modelo",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Número de imagens a gerar (padrão: 64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho para salvar imagens (padrão: ao lado do checkpoint)",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=8,
        help="Número de imagens por linha no grid (padrão: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo (cuda/cpu, padrão: auto-detectar)",
    )

    args = parser.parse_args()

    # Verificar se checkpoint existe
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint não encontrado: {args.checkpoint}")

    # Detectar dispositivo
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n🤖 Carregando modelo de: {args.checkpoint}")
    print(f"📱 Dispositivo: {device}")

    # Carregar checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    print(f"\n📋 Configurações do modelo:")
    print(f"   Dataset: {config.get('dataset', 'desconhecido')}")
    print(f"   Modelo: {config.get('model', 'desconhecido')}")
    print(f"   Época: {checkpoint.get('epoch', '?')}")
    print(
        f"   Tamanho da imagem: {config.get('img_size', 64)}x{config.get('img_size', 64)}"
    )
    print(f"   Canais: {config.get('nc', 3)}")

    # Criar modelo
    model_config = {
        "nz": config.get("nz", 100),
        "ngf": config.get("ngf", 64),
        "ndf": config.get("ndf", 64),
        "nc": config.get("nc", 3),
        "img_size": config.get("img_size", 64),
    }

    model_type = config.get("model", "dcgan")
    generator, _ = get_model(model_type, model_config)

    # Carregar pesos
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator = generator.to(device)
    generator.eval()

    print(f"\n✓ Modelo carregado com sucesso!")

    # Determinar caminho de saída
    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        parent_dir = os.path.dirname(checkpoint_dir)
        args.output = os.path.join(
            parent_dir, f"generated_{args.num_samples}_samples.png"
        )

    # Gerar imagens
    print(f"\n🎨 Gerando {args.num_samples} imagens...")

    nz = config.get("nz", 100)
    generate_samples(
        generator, args.num_samples, nz, device, args.output, nrow=args.nrow
    )

    print(f"\n✅ Imagens geradas e salvas em: {args.output}")
    print(f"   Total de imagens: {args.num_samples}")
    print(f"   Grid: {args.nrow} imagens por linha")
    print("\n✨ Concluído!\n")


if __name__ == "__main__":
    main()
