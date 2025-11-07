#!/usr/bin/env python3
"""
Script interativo para gerar imagens com seleção de classe/categoria

Este script permite:
1. Gerar imagens de classes específicas (ex: "gato" no CIFAR-10, "5" no MNIST)
2. Modo interativo com menu
3. Geração guiada por prompts

Nota: Para GANs incondicionais (sem labels no treinamento), as imagens são geradas
aleatoriamente. Para controle real de classes, é necessário treinar um Conditional GAN (c-GAN).

Uso:
    # Modo interativo
    python generate_interactive.py --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth

    # Especificar classe
    python generate_interactive.py --checkpoint outputs/mnist/dcgan_xxx/checkpoints/checkpoint_latest.pth --class-name "5"

    # Com prompt (simulado para GANs incondicionais)
    python generate_interactive.py --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth --prompt "gerar um gato"
"""

import argparse
import json
import os
import re

import torch

from config import DATASET_CONFIGS
from models import get_model
from utils import generate_samples


def parse_prompt(prompt, dataset_name):
    """
    Analisa um prompt de texto e extrai a classe desejada

    Args:
        prompt: Texto descrevendo o que gerar
        dataset_name: Nome do dataset

    Returns:
        classe extraída ou None
    """
    if dataset_name not in DATASET_CONFIGS:
        return None

    classes = DATASET_CONFIGS[dataset_name].get("classes", [])
    prompt_lower = prompt.lower()

    # Procurar por classes mencionadas no prompt
    for cls in classes:
        if cls.lower() in prompt_lower:
            return cls

    # Procurar por números (para MNIST)
    if dataset_name == "mnist":
        numbers = re.findall(r"\d", prompt)
        if numbers:
            return numbers[0]

    return None


def show_available_classes(dataset_name):
    """Mostra as classes disponíveis para um dataset"""
    if dataset_name not in DATASET_CONFIGS:
        print(f"⚠️  Dataset '{dataset_name}' não encontrado.")
        return []

    config = DATASET_CONFIGS[dataset_name]
    classes = config.get("classes", [])

    print(f"\n📦 Dataset: {config['name']}")
    print(f"🎯 Classes disponíveis ({len(classes)}):")
    print()

    for i, cls in enumerate(classes):
        print(f"  {i+1:2d}) {cls}")

    print()
    return classes


def interactive_menu(checkpoint_path, device):
    """Menu interativo para seleção de classe"""

    # Carregar checkpoint para obter informações
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    dataset_name = config.get("dataset", "unknown")

    print("\n" + "=" * 60)
    print("🎨 GERADOR INTERATIVO DE IMAGENS")
    print("=" * 60)

    # Mostrar classes disponíveis
    classes = show_available_classes(dataset_name)

    if not classes:
        print("⚠️  Este dataset não possui classes definidas.")
        print("    Gerando imagens aleatórias...")
        return None, None

    print("=" * 60)
    print("\n💡 IMPORTANTE:")
    print("   ⚠️  Este modelo foi treinado SEM labels (unconditional GAN)")
    print("   ⚠️  A seleção de classe é SIMULADA - gera imagens aleatórias")
    print("   ✅  Para controle real, treine um Conditional GAN (c-GAN)")
    print()
    print("=" * 60)
    print("\n🎯 Escolha uma das opções:")
    print()
    print("  1) Gerar classe específica (simulado)")
    print("  2) Gerar mistura de todas as classes")
    print("  3) Usar prompt de texto")
    print("  0) Cancelar")
    print()

    choice = input("Digite sua escolha [1-3]: ").strip()

    if choice == "0":
        return None, None

    elif choice == "1":
        # Selecionar classe específica
        print()
        class_num = input(f"Digite o número da classe [1-{len(classes)}]: ").strip()

        try:
            idx = int(class_num) - 1
            if 0 <= idx < len(classes):
                selected_class = classes[idx]
                print(f"\n✓ Selecionado: {selected_class}")
                print(f"  (Gerando imagens com tema '{selected_class}')")
                return selected_class, "specific"
            else:
                print("❌ Número inválido!")
                return None, None
        except ValueError:
            print("❌ Entrada inválida!")
            return None, None

    elif choice == "2":
        print("\n✓ Gerando mistura de todas as classes")
        return None, "mixed"

    elif choice == "3":
        # Prompt de texto
        print()
        print("💬 Digite o que você quer gerar:")
        print(f"   Exemplo: 'gerar um {classes[0].lower()}'")
        print()
        prompt = input("Prompt: ").strip()

        if not prompt:
            print("❌ Prompt vazio!")
            return None, None

        # Analisar prompt
        extracted_class = parse_prompt(prompt, dataset_name)

        if extracted_class:
            print(f"\n✓ Detectado: {extracted_class}")
            print(f"  Prompt: '{prompt}'")
            return extracted_class, "prompt"
        else:
            print(f"\n⚠️  Classe não identificada no prompt.")
            print(f"   Gerando imagens aleatórias...")
            return None, "prompt"

    else:
        print("❌ Opção inválida!")
        return None, None


def generate_with_class(
    generator, num_samples, nz, device, selected_class, dataset_name
):
    """
    Gera imagens 'temáticas' para uma classe

    Nota: Para GANs incondicionais, isso apenas gera ruído aleatório.
    Para controle real de classe, seria necessário um Conditional GAN.
    """
    print(f"\n🎨 Gerando {num_samples} imagens...")

    if selected_class:
        print(f"   🎯 Tema: {selected_class}")
        print(f"   ⚠️  Nota: Geração é aleatória (modelo incondicional)")
    else:
        print(f"   🎲 Modo: Aleatório (todas as classes)")

    # Gerar ruído aleatório
    # Para conditional GAN, aqui usaríamos embeddings de classe
    noise = torch.randn(num_samples, nz, 1, 1, device=device)

    with torch.no_grad():
        fake_images = generator(noise)

    return fake_images


def main():
    parser = argparse.ArgumentParser(
        description="Gerador interativo de imagens com seleção de classe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Modo interativo (recomendado)
  python generate_interactive.py --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth

  # Especificar classe diretamente
  python generate_interactive.py --checkpoint outputs/mnist/dcgan_xxx/checkpoints/checkpoint_latest.pth --class-name "5"

  # Usar prompt de texto
  python generate_interactive.py --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth --prompt "gerar um gato"

  # Gerar múltiplas imagens de uma classe
  python generate_interactive.py --checkpoint outputs/fashion-mnist/dcgan_xxx/checkpoints/checkpoint_latest.pth --class-name "Camiseta" --num-samples 16

Nota: Este script funciona melhor com modelos Conditional GAN (c-GAN).
Para GANs incondicionais, a seleção de classe é apenas simulada.
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Caminho para o checkpoint do modelo",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Nome da classe a gerar (ex: 'gato', '5', 'Camiseta')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt de texto descrevendo o que gerar",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Número de imagens a gerar (padrão: 16)",
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
        default=4,
        help="Número de imagens por linha no grid (padrão: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo (cuda/cpu, padrão: auto-detectar)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Desabilitar modo interativo",
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
    dataset_name = config.get("dataset", "unknown")

    print(f"\n📋 Configurações do modelo:")
    print(f"   Dataset: {dataset_name}")
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

    # Determinar classe a gerar
    selected_class = None
    mode = None

    if args.prompt:
        # Modo prompt
        selected_class = parse_prompt(args.prompt, dataset_name)
        mode = "prompt"
        print(f"\n💬 Prompt: '{args.prompt}'")
        if selected_class:
            print(f"✓ Classe detectada: {selected_class}")
        else:
            print(f"⚠️  Classe não detectada - gerando aleatoriamente")

    elif args.class_name:
        # Classe especificada diretamente
        selected_class = args.class_name
        mode = "specific"
        print(f"\n🎯 Classe selecionada: {selected_class}")

    elif not args.no_interactive:
        # Modo interativo
        selected_class, mode = interactive_menu(args.checkpoint, device)

        if mode is None:
            print("\n❌ Operação cancelada.")
            return

    # Gerar imagens
    fake_images = generate_with_class(
        generator,
        args.num_samples,
        model_config["nz"],
        device,
        selected_class,
        dataset_name,
    )

    # Determinar caminho de saída
    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        parent_dir = os.path.dirname(checkpoint_dir)

        if selected_class:
            class_safe = selected_class.replace(" ", "_").replace("/", "_")
            filename = f"generated_{class_safe}_{args.num_samples}.png"
        else:
            filename = f"generated_mixed_{args.num_samples}.png"

        args.output = os.path.join(parent_dir, filename)

    # Salvar imagens
    from torchvision.utils import save_image

    save_image(
        fake_images,
        args.output,
        nrow=args.nrow,
        normalize=True,
        value_range=(-1, 1),
    )

    print(f"\n✅ Imagens geradas e salvas em: {args.output}")
    print(f"   Total de imagens: {args.num_samples}")
    print(f"   Grid: {args.nrow} imagens por linha")

    if selected_class:
        print(f"   Tema: {selected_class}")

    print("\n" + "=" * 60)
    print("💡 DICA: Para controle real de classes:")
    print("   Treine um Conditional GAN (c-GAN) que usa labels durante o treinamento")
    print("=" * 60)

    print("\n✨ Concluído!\n")


if __name__ == "__main__":
    main()
