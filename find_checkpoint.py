#!/usr/bin/env python3
"""
Script auxiliar para encontrar checkpoints dinamicamente
Usado pelo INICIAR.bat para descobrir modelos instalados

Uso:
    python find_checkpoint.py mnist
    python find_checkpoint.py cifar10
    python find_checkpoint.py fashion-mnist
"""

import sys
import os
import glob
from pathlib import Path


def find_checkpoint_for_dataset(dataset_name):
    """
    Procura checkpoint_latest.pth para um dataset específico
    
    Args:
        dataset_name: Nome do dataset (mnist, cifar10, fashion-mnist)
        
    Returns:
        Caminho do checkpoint mais recente ou None se não encontrado
    """
    # Normalizar nome do dataset
    dataset_name = dataset_name.lower().strip()
    
    # Procurar em outputs/**/checkpoint_latest.pth
    checkpoints = []
    
    if not os.path.exists("outputs"):
        return None
    
    # Buscar recursivamente por checkpoint_latest.pth
    for ckpt_path in glob.glob(os.path.join("outputs", "**", "checkpoint_latest.pth"), recursive=True):
        # Normalizar separadores de caminho para Windows
        ckpt_path = ckpt_path.replace("/", os.sep)
        parts = ckpt_path.split(os.sep)
        
        # Estrutura esperada: outputs/<dataset>/...
        if len(parts) >= 3 and parts[0] == "outputs":
            ds = parts[1].lower()
            
            # Verificar se o dataset corresponde
            if ds == dataset_name:
                try:
                    mtime = os.path.getmtime(ckpt_path)
                    checkpoints.append((ckpt_path, mtime))
                except OSError:
                    continue
    
    # Se encontrou checkpoints, retornar o mais recente
    if checkpoints:
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]
    
    return None


def main():
    if len(sys.argv) < 2:
        print("Uso: python find_checkpoint.py <dataset>", file=sys.stderr)
        print("Exemplo: python find_checkpoint.py mnist", file=sys.stderr)
        sys.exit(1)
    
    dataset = sys.argv[1]
    checkpoint = find_checkpoint_for_dataset(dataset)
    
    if checkpoint:
        # Imprimir o caminho encontrado (stdout)
        print(checkpoint)
        sys.exit(0)
    else:
        # Não encontrado
        sys.exit(1)


if __name__ == "__main__":
    main()
