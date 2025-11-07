#!/usr/bin/env python3
"""
Script de teste para verificar se tudo está funcionando
"""

import os
import sys


def test_imports():
    """Testa se todas as importações funcionam"""
    print("🔍 Testando importações...")

    try:
        import torch

        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch não encontrado: {e}")
        return False

    try:
        import torchvision

        print(f"  ✅ torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"  ❌ torchvision não encontrado: {e}")
        return False

    try:
        import matplotlib

        print(f"  ✅ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ❌ matplotlib não encontrado: {e}")
        return False

    try:
        from models import get_model

        print("  ✅ models.py")
    except ImportError as e:
        print(f"  ❌ Erro em models.py: {e}")
        return False

    try:
        from config import get_dataset, get_model_config

        print("  ✅ config.py")
    except ImportError as e:
        print(f"  ❌ Erro em config.py: {e}")
        return False

    try:
        from utils import TrainingLogger, get_device

        print("  ✅ utils.py")
    except ImportError as e:
        print(f"  ❌ Erro em utils.py: {e}")
        return False

    return True


def test_gpu():
    """Testa disponibilidade de GPU"""
    print("\n🎮 Testando GPU...")

    import torch

    if torch.cuda.is_available():
        print(f"  ✅ GPU disponível: {torch.cuda.get_device_name(0)}")
        print(
            f"  📊 Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        return True
    else:
        print("  ⚠️  GPU não disponível (usará CPU)")
        return False


def test_model_creation():
    """Testa criação de modelos"""
    print("\n🤖 Testando criação de modelos...")

    try:
        from models import get_model

        # Testar DCGAN
        model_config = {
            "nz": 100,
            "ngf": 64,
            "ndf": 64,
            "nc": 3,
            "img_size": 64,
        }

        generator, discriminator = get_model("dcgan", model_config)
        print("  ✅ DCGAN criado com sucesso")

        # Testar WGAN-GP
        generator, critic = get_model("wgan-gp", model_config)
        print("  ✅ WGAN-GP criado com sucesso")

        return True
    except Exception as e:
        print(f"  ❌ Erro ao criar modelos: {e}")
        return False


def test_dataset_config():
    """Testa configuração de datasets"""
    print("\n📦 Testando configuração de datasets...")

    try:
        from config import DATASET_CONFIGS, MODEL_CONFIGS

        print(f"  ✅ Datasets disponíveis: {len(DATASET_CONFIGS)}")
        for name in DATASET_CONFIGS:
            print(f"     • {name}")

        print(f"  ✅ Modelos disponíveis: {len(MODEL_CONFIGS)}")
        for name in MODEL_CONFIGS:
            print(f"     • {name}")

        return True
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        return False


def test_file_structure():
    """Testa se todos os arquivos necessários existem"""
    print("\n📁 Testando estrutura de arquivos...")

    required_files = [
        "train.py",
        "generate.py",
        "models.py",
        "config.py",
        "utils.py",
        "requirements.txt",
        "TRAINING_GUIDE.md",
    ]

    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} não encontrado")
            all_exist = False

    return all_exist


def main():
    print("=" * 70)
    print("🧪 TESTE DO SISTEMA DE TREINAMENTO DE GANs")
    print("=" * 70)

    results = []

    # Executar testes
    results.append(("Estrutura de arquivos", test_file_structure()))
    results.append(("Importações", test_imports()))
    results.append(("GPU", test_gpu()))
    results.append(("Criação de modelos", test_model_creation()))
    results.append(("Configuração de datasets", test_dataset_config()))

    # Resumo
    print("\n" + "=" * 70)
    print("📊 RESUMO DOS TESTES")
    print("=" * 70)

    all_passed = True
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:.<40} {status}")
        if not result:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("\n✨ Sistema pronto para uso!")
        print("\nPróximos passos:")
        print("  1. Ver datasets: python train.py --list-datasets")
        print("  2. Ver modelos: python train.py --list-models")
        print("  3. Treinar: python train.py --dataset mnist --model dcgan --epochs 5")
        return 0
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM")
        print("\nPor favor:")
        print("  1. Instale as dependências: pip install -r requirements.txt")
        print("  2. Verifique os arquivos faltantes")
        print("  3. Execute novamente: python test_system.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
