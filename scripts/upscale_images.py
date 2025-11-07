#!/usr/bin/env python3
"""
Script para fazer upscaling de imagens geradas por GANs

Usa técnicas de interpolação e super-resolução para aumentar
a resolução de imagens geradas, mantendo qualidade visual.

Métodos disponíveis:
1. Bicubic: Rápido, resultados bons
2. Lanczos: Melhor qualidade, um pouco mais lento
3. ESRGAN (opcional): Melhor qualidade, requer modelo extra

Uso:
    # Upscale simples (bicubic)
    python upscale_images.py --input generated.png --scale 4 --output generated_hd.png

    # Melhor qualidade (lanczos)
    python upscale_images.py --input generated.png --scale 8 --method lanczos

    # Super-resolução com ESRGAN (requer modelo)
    python upscale_images.py --input generated.png --method esrgan --scale 4
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def upscale_bicubic(image, scale_factor):
    """Upscaling usando interpolação bicúbica (rápido)"""
    width, height = image.size
    new_size = (width * scale_factor, height * scale_factor)
    return image.resize(new_size, Image.BICUBIC)


def upscale_lanczos(image, scale_factor):
    """Upscaling usando Lanczos (melhor qualidade)"""
    width, height = image.size
    new_size = (width * scale_factor, height * scale_factor)
    return image.resize(new_size, Image.LANCZOS)


def upscale_nearest(image, scale_factor):
    """Upscaling pixel-perfect (estilo retro)"""
    width, height = image.size
    new_size = (width * scale_factor, height * scale_factor)
    return image.resize(new_size, Image.NEAREST)


def enhance_sharpness(image, factor=1.5):
    """Aumenta a nitidez da imagem"""
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def enhance_contrast(image, factor=1.2):
    """Melhora o contraste"""
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def upscale_esrgan(image, scale_factor):
    """
    Upscaling com ESRGAN (super-resolução com deep learning)

    Nota: Requer modelo ESRGAN baixado.
    Para usar: pip install basicsr realesrgan
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        print("❌ ESRGAN não instalado!")
        print("   Instale com: pip install basicsr realesrgan")
        print("   Usando bicubic como fallback...")
        return upscale_bicubic(image, scale_factor)

    try:
        # Modelo ESRGAN (baixa automaticamente se necessário)
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

        upsampler = RealESRGANer(
            scale=4,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )

        # Converter para numpy array
        img_array = np.array(image)

        # Upscale
        output, _ = upsampler.enhance(img_array, outscale=scale_factor)

        # Converter de volta para PIL
        return Image.fromarray(output)

    except Exception as e:
        print(f"⚠️  Erro ao usar ESRGAN: {e}")
        print("   Usando bicubic como fallback...")
        return upscale_bicubic(image, scale_factor)


def main():
    parser = argparse.ArgumentParser(
        description="Upscaling de imagens geradas por GANs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Upscale 4x com bicubic (rápido)
  python upscale_images.py --input outputs/mnist/generated.png --scale 4
  
  # Upscale 8x com lanczos (melhor qualidade)
  python upscale_images.py --input outputs/cifar10/generated.png --scale 8 --method lanczos
  
  # Upscale com nitidez aumentada
  python upscale_images.py --input generated.png --scale 4 --sharpen 1.8
  
  # Estilo pixel-art (nearest neighbor)
  python upscale_images.py --input generated.png --scale 4 --method nearest
  
  # Super-resolução com ESRGAN (melhor qualidade, mais lento)
  python upscale_images.py --input generated.png --method esrgan --scale 4

Recomendações:
  - MNIST 28x28 → 224x224: --scale 8 --method lanczos
  - CIFAR-10 32x32 → 256x256: --scale 8 --method lanczos
  - Fashion-MNIST 28x28 → 280x280: --scale 10 --method lanczos
        """,
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Caminho da imagem de entrada"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Caminho da imagem de saída (padrão: input_upscaled_NNx.png)",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=int,
        default=4,
        help="Fator de escala (2, 4, 8, etc.) (padrão: 4)",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        choices=["bicubic", "lanczos", "nearest", "esrgan"],
        default="bicubic",
        help="Método de upscaling (padrão: bicubic)",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=None,
        help="Fator de nitidez (1.0-2.0, padrão: desabilitado)",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=None,
        help="Fator de contraste (1.0-1.5, padrão: desabilitado)",
    )

    args = parser.parse_args()

    # Verificar se arquivo existe
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Arquivo não encontrado: {args.input}")

    print(f"\n🖼️  Carregando imagem: {args.input}")

    # Carregar imagem
    image = Image.open(args.input)
    original_size = image.size

    print(f"   Tamanho original: {original_size[0]}x{original_size[1]}")
    print(f"   Modo: {image.mode}")

    # Converter para RGB se necessário
    if image.mode != "RGB":
        print(f"   Convertendo {image.mode} → RGB")
        image = image.convert("RGB")

    # Upscale
    print(f"\n🚀 Aplicando upscale {args.scale}x usando {args.method}...")

    if args.method == "bicubic":
        upscaled = upscale_bicubic(image, args.scale)
    elif args.method == "lanczos":
        upscaled = upscale_lanczos(image, args.scale)
    elif args.method == "nearest":
        upscaled = upscale_nearest(image, args.scale)
    elif args.method == "esrgan":
        upscaled = upscale_esrgan(image, args.scale)
    else:
        upscaled = upscale_bicubic(image, args.scale)

    new_size = upscaled.size
    print(f"   Novo tamanho: {new_size[0]}x{new_size[1]}")

    # Aplicar melhorias opcionais
    if args.sharpen is not None:
        print(f"\n✨ Aumentando nitidez (fator: {args.sharpen})...")
        upscaled = enhance_sharpness(upscaled, args.sharpen)

    if args.contrast is not None:
        print(f"✨ Ajustando contraste (fator: {args.contrast})...")
        upscaled = enhance_contrast(upscaled, args.contrast)

    # Determinar caminho de saída
    if args.output is None:
        input_path = Path(args.input)
        stem = input_path.stem
        suffix = input_path.suffix
        args.output = str(input_path.parent / f"{stem}_upscaled_{args.scale}x{suffix}")

    # Salvar
    print(f"\n💾 Salvando em: {args.output}")
    upscaled.save(args.output, quality=95)

    # Estatísticas
    original_file_size = os.path.getsize(args.input) / 1024
    new_file_size = os.path.getsize(args.output) / 1024

    print(f"\n✅ Concluído!")
    print(
        f"   Resolução: {original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]}"
    )
    print(f"   Escala: {args.scale}x")
    print(f"   Método: {args.method}")
    print(f"   Tamanho: {original_file_size:.1f}KB → {new_file_size:.1f}KB")

    print("\n" + "=" * 60)
    print("💡 DICAS:")
    print("   • Lanczos geralmente dá melhor qualidade que bicubic")
    print("   • Nearest é bom para pixel-art/sprites")
    print("   • ESRGAN dá a melhor qualidade (mas é mais lento)")
    print("   • Sharpen 1.5-1.8 melhora a percepção de qualidade")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
