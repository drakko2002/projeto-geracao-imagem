#!/usr/bin/env python3
"""
Script para testar detecção de GPU
"""
import torch

print("=" * 70)
print("TESTE DE DETECÇÃO DE GPU")
print("=" * 70)

print(f"\n✓ PyTorch Version: {torch.__version__}")
print(f"✓ CUDA disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ Número de GPUs: {torch.cuda.device_count()}")
    print(f"✓ GPU Atual: {torch.cuda.current_device()}")
    print(f"✓ Nome da GPU: {torch.cuda.get_device_name(0)}")

    # Teste simples de alocação
    print("\nTestando alocação na GPU...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = x @ y
        print("✓ Teste de multiplicação de matrizes na GPU: SUCESSO")
        print(f"✓ Dispositivo do tensor: {z.device}")
    except Exception as e:
        print(f"✗ Erro ao testar GPU: {e}")
else:
    print("\n⚠ CUDA não está disponível!")
    print("\nPossíveis causas:")
    print("  1. PyTorch instalado sem suporte CUDA")
    print("  2. Driver NVIDIA não instalado/configurado corretamente")
    print("  3. Versão do CUDA incompatível")
    print("\nPara instalar PyTorch com CUDA:")
    print(
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    )

print("\n" + "=" * 70)
