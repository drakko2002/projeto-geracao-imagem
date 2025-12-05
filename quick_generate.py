#!/usr/bin/env python3
"""
Script auxiliar para gerar imagens rapidamente
Encontra automaticamente o último modelo treinado
"""

import glob
import os
import subprocess
import sys


def find_latest_checkpoint():
    """Encontra o checkpoint mais recente"""
    # Procurar todos os checkpoints
    checkpoints = glob.glob("outputs/**/checkpoint_latest.pth", recursive=True)

    if not checkpoints:
        print("❌ Nenhum checkpoint encontrado!")
        print("\nVocê precisa treinar um modelo primeiro:")
        print("  python train.py --dataset mnist --model dcgan --epochs 5")
        return None

    # Ordenar por data de modificação (mais recente primeiro)
    checkpoints.sort(key=os.path.getmtime, reverse=True)

    return checkpoints[0]


def main():
    print("🔍 Procurando modelos treinados...\n")

    checkpoint = find_latest_checkpoint()

    if checkpoint is None:
        return 1

    print(f"✅ Checkpoint encontrado: {checkpoint}")
    print(f"   Modificado em: {os.path.getmtime(checkpoint)}")

    # Extrair informações do caminho
    parts = checkpoint.split(os.sep)
    dataset = parts[1] if len(parts) > 1 else "unknown"
    model_dir = parts[2] if len(parts) > 2 else "unknown"

    print(f"\n📊 Informações:")
    print(f"   Dataset: {dataset}")
    print(f"   Modelo: {model_dir}")

    # Perguntar quantas imagens gerar
    print("\n" + "=" * 60)
    num_samples = input("Quantas imagens gerar? (padrão: 64): ").strip()
    if not num_samples:
        num_samples = "64"
    
    # Perguntar sobre upscale
    upscale = input("Aplicar upscaling? (none/2x/4x/8x, padrão: none): ").strip().lower()
    if not upscale or upscale not in ["2x", "4x", "8x"]:
        upscale = "none"

    # Montar comando
    cmd = [
        "python",
        "generate.py",
        "--checkpoint",
        checkpoint,
        "--num-samples",
        num_samples,
        "--upscale",
        upscale,
    ]

    print(f"\n🎨 Gerando {num_samples} imagens...")
    if upscale != "none":
        print(f"   Com upscaling {upscale}")
    print(f"Comando: {' '.join(cmd)}\n")

    # Executar
    result = subprocess.run(cmd)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
