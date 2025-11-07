#!/usr/bin/env python3
"""
Script para treinar DCGAN com diferentes datasets
Suporta CIFAR-10, MNIST, Fashion-MNIST e outros
"""

import argparse
import os
import subprocess
import sys


def get_dataset_info(dataset):
    datasets = {
        "cifar10": {
            "description": "Imagens coloridas de 10 categorias:",
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
            "channels": 3,  # RGB
        },
        "mnist": {
            "description": "Dígitos escritos à mão de 0-9:",
            "classes": [str(i) for i in range(10)],
            "channels": 1,  # Escala de cinza
        },
        "fashion-mnist": {
            "description": "Imagens de roupas e acessórios:",
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
            "channels": 1,  # Escala de cinza
        },
    }
    return datasets.get(dataset, None)


def main():
    parser = argparse.ArgumentParser(
        description="Treinamento de DCGAN com diferentes datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "mnist", "fashion-mnist"],
        help="Dataset para treinamento",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Número de épocas de treinamento"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do batch")
    parser.add_argument(
        "--image-size", type=int, default=64, help="Tamanho das imagens (quadradas)"
    )
    parser.add_argument("--ngpu", type=int, default=1, help="Número de GPUs para usar")

    args = parser.parse_args()

    # Configurações
    DATASET = args.dataset
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size
    OUTPUT_DIR = f"outputs/{DATASET}"
    DATA_DIR = "./data"

    dataset_info = get_dataset_info(DATASET)
    if dataset_info is None:
        print(f"Dataset '{DATASET}' não suportado!")
        return

    print("=" * 70)
    print(f"TREINAMENTO DCGAN - {DATASET.upper()}")
    print("=" * 70)
    print(f"\n{dataset_info['description']}")
    for i, classe in enumerate(dataset_info["classes"], 1):
        print(f"  • {classe}")
    print("\nO modelo aprenderá a gerar imagens similares a essas categorias.")
    print("=" * 70)

    print(f"\nConfigurações:")
    print(f"  • Dataset: {DATASET}")
    print(f"  • Épocas: {EPOCHS}")
    print(f"  • Batch Size: {BATCH_SIZE}")
    print(f"  • Tamanho da Imagem: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  • Diretório de Saída: {OUTPUT_DIR}")
    print(f"  • Diretório de Dados: {DATA_DIR}")

    # Criar diretórios se não existirem
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("Iniciando treinamento...")
    print("(Isso pode demorar alguns minutos ou horas dependendo do hardware)")
    print("=" * 70 + "\n")

    cmd = [
        sys.executable,
        "dcgan/main.py",
        "--dataset",
        DATASET,
        "--dataroot",
        DATA_DIR,
        "--niter",
        str(EPOCHS),
        "--batchSize",
        str(BATCH_SIZE),
        "--imageSize",
        str(IMAGE_SIZE),
        "--outf",
        OUTPUT_DIR,
        "--ngpu",
        str(args.ngpu),
    ]

    try:
        import torch

        if torch.cuda.is_available():
            print("✓ GPU detectada! Usando aceleração CUDA")
        else:
            print("ℹ GPU não detectada. Usando CPU (será mais lento)")
    except:
        pass

    print("\nExecutando comando:")
    print(" ".join(cmd))
    print("\n" + "=" * 70 + "\n")

    try:
        # Executar o treinamento
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("✓ TREINAMENTO CONCLUÍDO COM SUCESSO!")
            print("=" * 70)
            print(f"\nImagens geradas salvas em: {OUTPUT_DIR}/")
            print(f"Modelos salvos em: {OUTPUT_DIR}/")
            print("\nArquivos gerados:")
            print(f"  • real_samples.png - Exemplos de imagens reais do dataset")
            print(f"  • fake_samples_epoch_*.png - Imagens geradas em cada época")
            print(f"  • netG_epoch_*.pth - Pesos do gerador")
            print(f"  • netD_epoch_*.pth - Pesos do discriminador")
        else:
            print("\n✗ Treinamento interrompido ou com erro")

    except KeyboardInterrupt:
        print("\n\n⚠ Treinamento interrompido pelo usuário (Ctrl+C)")
        print("Os modelos parcialmente treinados foram salvos.")
    except Exception as e:
        print(f"\n✗ Erro durante o treinamento: {e}")


if __name__ == "__main__":
    main()
