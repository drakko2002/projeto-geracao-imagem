#!/usr/bin/env python3
"""
Script para treinar DCGAN com dataset CIFAR-10
Dataset com 60.000 imagens coloridas de 10 categorias diferentes
"""

import os
import subprocess
import sys

print("=" * 70)
print("TREINAMENTO DCGAN - CIFAR-10")
print("=" * 70)
print("\nO dataset CIFAR-10 contém imagens de:")
print("  • Aviões, carros, pássaros, gatos, cervos")
print("  • Cachorros, sapos, cavalos, navios, caminhões")
print("\nO modelo aprenderá a gerar imagens similares a essas categorias.")
print("=" * 70)

# Configurações
EPOCHS = 25  # Número de épocas (25 é bom para ver resultados)
BATCH_SIZE = 64
IMAGE_SIZE = 64
OUTPUT_DIR = "outputs/cifar10"
DATA_DIR = "./data"

print(f"\nConfigurações:")
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

# Comando para executar o treinamento
cmd = [
    sys.executable,
    "dcgan/main.py",
    "--dataset",
    "cifar10",
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
    "1",  # Ajuste se tiver mais GPUs
]

# Se tiver CUDA disponível, use
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

print("\n" + "=" * 70)
