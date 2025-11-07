#!/usr/bin/env python3
"""
Script interativo para gerar imagens com seleção de classe/categoria

Este script permite:
1. Gerar imagens de classes específicas (ex: "gato" no CIFAR-10, "5" no MNIST)
2. Modo interativo com menu
3. Geração guiada por prompts
4. Upscaling automático para alta resolução

Nota: Para GANs incondicionais (sem labels no treinamento), as imagens são geradas
aleatoriamente. Para controle real de classes, é necessário treinar um Conditional GAN (c-GAN).

Uso:
    # Modo interativo (gera 1 imagem em alta resolução)
    python generate_interactive.py --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth

    # Especificar classe
    python generate_interactive.py --checkpoint outputs/mnist/dcgan_xxx/checkpoints/checkpoint_latest.pth --class-name "5"

    # Com prompt (simulado para GANs incondicionais)
    python generate_interactive.py --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth --prompt "gerar um gato"

    # Desabilitar upscaling
    python generate_interactive.py --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth --upscale 1

    # Gerar múltiplas imagens
    python generate_interactive.py --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth --num-samples 16
"""

import argparse
import json
import os
import re

import numpy as np
import torch
from PIL import Image, ImageEnhance

from config import DATASET_CONFIGS
from models import get_model
from utils import generate_samples


def upscale_image(image_tensor, scale_factor, method="lanczos", sharpen=1.0):
    """
    Faz upscaling de um tensor de imagem

    Args:
        image_tensor: Tensor PyTorch (C, H, W) normalizado em [-1, 1]
        scale_factor: Fator de escala (2, 4, 8, etc)
        method: Método de interpolação ('lanczos', 'bicubic', 'nearest')
        sharpen: Fator de nitidez (1.0 = sem alteração, >1.0 = mais nítido)

    Returns:
        Tensor upscaled no mesmo formato
    """
    if scale_factor == 1:
        return image_tensor

    # Converter tensor para PIL Image
    # Desnormalizar: [-1, 1] -> [0, 1]
    img_np = ((image_tensor + 1) / 2).clamp(0, 1).cpu().numpy()

    # Converter de (C, H, W) para (H, W, C)
    img_np = np.transpose(img_np, (1, 2, 0))

    # Converter para uint8
    img_np = (img_np * 255).astype(np.uint8)

    # Converter para PIL
    if img_np.shape[2] == 1:
        # Grayscale
        pil_image = Image.fromarray(img_np[:, :, 0], mode="L")
    else:
        # RGB
        pil_image = Image.fromarray(img_np, mode="RGB")

    # Aplicar upscaling
    width, height = pil_image.size
    new_size = (width * scale_factor, height * scale_factor)

    if method == "lanczos":
        upscaled = pil_image.resize(new_size, Image.LANCZOS)
    elif method == "bicubic":
        upscaled = pil_image.resize(new_size, Image.BICUBIC)
    elif method == "nearest":
        upscaled = pil_image.resize(new_size, Image.NEAREST)
    else:
        upscaled = pil_image.resize(new_size, Image.LANCZOS)

    # Aplicar nitidez
    if sharpen > 1.0:
        enhancer = ImageEnhance.Sharpness(upscaled)
        upscaled = enhancer.enhance(sharpen)

    # Converter de volta para tensor
    img_np = np.array(upscaled).astype(np.float32) / 255.0

    # Adicionar dimensão de canal se necessário (grayscale)
    if len(img_np.shape) == 2:
        img_np = img_np[:, :, np.newaxis]

    # Converter de (H, W, C) para (C, H, W)
    img_np = np.transpose(img_np, (2, 0, 1))

    # Normalizar de volta para [-1, 1]
    img_tensor = torch.from_numpy(img_np) * 2 - 1

    return img_tensor


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
        default=1,
        help="Número de imagens a gerar (padrão: 1)",
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
    parser.add_argument(
        "--upscale",
        type=int,
        default=8,
        help="Fator de upscaling automático (padrão: 8x, use 1 para desabilitar)",
    )
    parser.add_argument(
        "--upscale-method",
        type=str,
        default="lanczos",
        choices=["lanczos", "bicubic", "nearest"],
        help="Método de upscaling (padrão: lanczos)",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=1.6,
        help="Fator de nitidez no upscaling (1.0-2.0, padrão: 1.6, use 1.0 para desabilitar)",
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
    print(f"\n🎨 Gerando {args.num_samples} imagem(ns)...")

    fake_images = generate_with_class(
        generator,
        args.num_samples,
        model_config["nz"],
        device,
        selected_class,
        dataset_name,
    )

    original_size = fake_images.shape[-1]  # Altura/largura original

    # Aplicar upscaling se necessário
    if args.upscale > 1:
        print(
            f"\n📐 Aplicando upscaling {args.upscale}x ({original_size}x{original_size} → {original_size * args.upscale}x{original_size * args.upscale})..."
        )
        print(f"   Método: {args.upscale_method}")
        if args.sharpen > 1.0:
            print(f"   Nitidez: {args.sharpen}")

        upscaled_images = []
        for i in range(fake_images.shape[0]):
            upscaled = upscale_image(
                fake_images[i],
                args.upscale,
                method=args.upscale_method,
                sharpen=args.sharpen,
            )
            upscaled_images.append(upscaled)

        fake_images = torch.stack(upscaled_images)
        final_size = original_size * args.upscale
    else:
        final_size = original_size

    # Determinar caminho de saída
    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        parent_dir = os.path.dirname(checkpoint_dir)

        if selected_class:
            class_safe = selected_class.replace(" ", "_").replace("/", "_")
            base_filename = f"generated_{class_safe}"
        else:
            base_filename = f"generated"

        # Adicionar informação de upscale no nome
        if args.upscale > 1:
            base_filename += f"_{final_size}x{final_size}"

        args.output = os.path.join(parent_dir, base_filename)

    # Salvar imagens
    from torchvision.utils import save_image

    print(f"\n💾 Salvando imagem(ns)...")

    if args.num_samples == 1:
        # Salvar imagem única
        output_path = (
            args.output if args.output.endswith(".png") else args.output + ".png"
        )
        save_image(
            fake_images[0],
            output_path,
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"\n✅ Imagem gerada e salva em: {output_path}")
        print(f"   Resolução: {final_size}x{final_size}")
        if selected_class:
            print(f"   Tema: {selected_class}")
        if args.upscale > 1:
            print(
                f"   Upscaling: {args.upscale}x ({original_size}x{original_size} → {final_size}x{final_size})"
            )
            print(f"   Método: {args.upscale_method}")
    else:
        # Salvar grid de múltiplas imagens
        grid_path = (
            args.output if args.output.endswith(".png") else args.output + "_grid.png"
        )
        save_image(
            fake_images,
            grid_path,
            nrow=args.nrow,
            normalize=True,
            value_range=(-1, 1),
        )

        # Também salvar individualmente
        output_dir = args.output + "_individual"
        os.makedirs(output_dir, exist_ok=True)

        for i in range(args.num_samples):
            individual_path = os.path.join(output_dir, f"image_{i+1:03d}.png")
            save_image(
                fake_images[i],
                individual_path,
                normalize=True,
                value_range=(-1, 1),
            )

        print(f"\n✅ Imagens geradas e salvas:")
        print(f"   Grid: {grid_path}")
        print(f"   Individuais: {output_dir}/ (1-{args.num_samples})")
        print(f"   Resolução: {final_size}x{final_size}")
        print(f"   Total: {args.num_samples} imagens")
        if selected_class:
            print(f"   Tema: {selected_class}")
        if args.upscale > 1:
            print(
                f"   Upscaling: {args.upscale}x ({original_size}x{original_size} → {final_size}x{final_size})"
            )

    print("\n" + "=" * 70)
    if args.upscale > 1:
        print("✨ Imagem gerada em ALTA RESOLUÇÃO com upscaling automático!")
    print("💡 DICAS:")
    print("   • Use --upscale 1 para desabilitar upscaling")
    print("   • Use --upscale-method esrgan para máxima qualidade (requer instalação)")
    print("   • Use --sharpen 1.0 para desabilitar aumento de nitidez")
    print("   • Para controle real de classes, treine um Conditional GAN (c-GAN)")
    print("=" * 70)

    print("\n✨ Concluído!\n")


if __name__ == "__main__":
    main()
