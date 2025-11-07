#!/usr/bin/env python3
"""Teste rápido das arquiteturas"""

import torch

from models import get_model


def test_architecture(model_type, img_size, nc):
    """Testa se a arquitetura funciona com tamanho específico"""
    print(f"\n{'='*60}")
    print(f"Testando {model_type.upper()} com img_size={img_size}, nc={nc}")
    print(f"{'='*60}")

    model_config = {
        "nz": 100,
        "ngf": 64,
        "ndf": 64,
        "nc": nc,
        "img_size": img_size,
    }

    try:
        generator, discriminator = get_model(model_type, model_config)

        # Teste do gerador
        noise = torch.randn(4, 100, 1, 1)
        fake_images = generator(noise)
        print(f"✅ Gerador: {noise.shape} -> {fake_images.shape}")
        assert fake_images.shape == (
            4,
            nc,
            img_size,
            img_size,
        ), f"Esperado (4, {nc}, {img_size}, {img_size}), obtido {fake_images.shape}"

        # Teste do discriminador
        real_images = torch.randn(4, nc, img_size, img_size)
        output = discriminator(real_images)
        print(f"✅ Discriminador: {real_images.shape} -> {output.shape}")
        assert output.shape == (4,), f"Esperado (4,), obtido {output.shape}"

        print(f"✅ Todos os testes passaram!")
        return True

    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback

        traceback.print_exc()
        return False


# Testar diferentes configurações
configs = [
    ("dcgan", 64, 3),  # CIFAR-10
    ("dcgan", 64, 1),  # MNIST
    ("dcgan", 32, 3),  # Menor
    ("dcgan", 128, 3),  # Maior
    ("wgan-gp", 64, 3),
    ("wgan-gp", 64, 1),
]

results = []
for model_type, img_size, nc in configs:
    success = test_architecture(model_type, img_size, nc)
    results.append((model_type, img_size, nc, success))

# Resumo
print(f"\n{'='*60}")
print("RESUMO DOS TESTES")
print(f"{'='*60}")
for model_type, img_size, nc, success in results:
    status = "✅" if success else "❌"
    print(f"{status} {model_type} | img_size={img_size} | nc={nc}")

all_passed = all(r[3] for r in results)
if all_passed:
    print(f"\n🎉 Todos os testes passaram!")
else:
    print(f"\n⚠️ Alguns testes falharam")
