#!/usr/bin/env python3
"""
Script para baixar modelos pré-treinados do Google Drive

Este script baixa checkpoints de modelos já treinados para que você possa
testar o gerador de imagens sem precisar treinar do zero.

Uso:
    python download_models.py --all                    # Baixar todos os modelos
    python download_models.py --model mnist            # Baixar apenas MNIST
    python download_models.py --model cifar10          # Baixar apenas CIFAR-10
    python download_models.py --model fashion-mnist    # Baixar apenas Fashion-MNIST
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    print("❌ Biblioteca 'gdown' não encontrada!")
    print("   Instalando...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown


# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DOS MODELOS DISPONÍVEIS
# ══════════════════════════════════════════════════════════════

MODELS = {
    "mnist": {
        "name": "MNIST DCGAN",
        "description": "Modelo treinado em dígitos 0-9 (28x28)",
        "epochs": 25,
        "size": "~50MB",
        "google_drive_id": "104QtE6vFOZn7euCORSZYjOvDN_mFHytq",
        "output_dir": "outputs/mnist/dcgan_pretrained",
    },
    "fashion-mnist": {
        "name": "Fashion-MNIST DCGAN",
        "description": "Modelo treinado em roupas e acessórios (28x28)",
        "epochs": 30,
        "size": "~50MB",
        "google_drive_id": "SEU_ID_AQUI",
        "output_dir": "outputs/fashion-mnist/dcgan_pretrained",
    },
    "cifar10": {
        "name": "CIFAR-10 DCGAN",
        "description": "Modelo treinado em 10 classes coloridas (32x32)",
        "epochs": 50,
        "size": "~80MB",
        "google_drive_id": "17mTNwrz8n6YWqnkfn7tuWQsX9s3s5ujO",
        "output_dir": "outputs/cifar10/dcgan_pretrained",
    },
}


# ══════════════════════════════════════════════════════════════
# FUNÇÕES
# ══════════════════════════════════════════════════════════════


def download_model(model_key):
    """
    Baixa um modelo específico do Google Drive

    Args:
        model_key: Chave do modelo (mnist, cifar10, fashion-mnist)
    """
    if model_key not in MODELS:
        print(f"❌ Modelo '{model_key}' não encontrado!")
        print(f"   Modelos disponíveis: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_key]

    print("\n" + "=" * 70)
    print(f"📦 Baixando: {model_info['name']}")
    print("=" * 70)
    print(f"   Descrição: {model_info['description']}")
    print(f"   Épocas treinadas: {model_info['epochs']}")
    print(f"   Tamanho: {model_info['size']}")
    print(f"   Destino: {model_info['output_dir']}")
    print()

    # Verificar se ID do Google Drive foi configurado
    if model_info["google_drive_id"] == "SEU_ID_AQUI":
        print("⚠️  Este modelo ainda não está disponível para download!")
        print("   O ID do Google Drive precisa ser configurado.")
        print()
        print("   INSTRUÇÕES:")
        print("   1. Faça upload do modelo treinado para o Google Drive")
        print("   2. Compartilhe o arquivo (qualquer pessoa com o link)")
        print("   3. Copie o ID do arquivo (da URL do Google Drive)")
        print("   4. Atualize o arquivo download_models.py com o ID")
        print()
        return False

    # Criar diretório de destino
    output_dir = Path(model_info["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Caminho do arquivo zip temporário
    zip_path = output_dir / f"{model_key}_checkpoint.zip"

    # Baixar do Google Drive
    try:
        print("📥 Baixando do Google Drive...")
        url = f"https://drive.google.com/uc?id={model_info['google_drive_id']}"
        gdown.download(url, str(zip_path), quiet=False)

        # Extrair arquivo
        print("\n📂 Extraindo arquivo...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        # Remover arquivo zip
        zip_path.unlink()

        print(f"\n✅ Modelo '{model_info['name']}' baixado com sucesso!")
        print(f"   Localização: {output_dir}")
        print()

        return True

    except Exception as e:
        print(f"\n❌ Erro ao baixar modelo: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


def list_models():
    """Lista todos os modelos disponíveis"""
    print("\n" + "=" * 70)
    print("📦 MODELOS PRÉ-TREINADOS DISPONÍVEIS")
    print("=" * 70)
    print()

    for key, info in MODELS.items():
        status = (
            "✅ Configurado"
            if info["google_drive_id"] != "SEU_ID_AQUI"
            else "⚠️  Pendente"
        )

        print(f"🤖 {info['name']}")
        print(f"   Dataset: {key}")
        print(f"   Descrição: {info['description']}")
        print(f"   Épocas: {info['epochs']} | Tamanho: {info['size']}")
        print(f"   Status: {status}")
        print()


def check_existing_models():
    """Verifica quais modelos já estão baixados"""
    print("\n" + "=" * 70)
    print("📂 VERIFICANDO MODELOS LOCAIS")
    print("=" * 70)
    print()

    found_models = []

    for key, info in MODELS.items():
        output_dir = Path(info["output_dir"])
        checkpoints_dir = output_dir / "checkpoints"

        if checkpoints_dir.exists() and any(checkpoints_dir.glob("*.pth")):
            print(f"✅ {info['name']}: ENCONTRADO")
            print(f"   Localização: {checkpoints_dir}")

            # Listar checkpoints disponíveis
            checkpoints = sorted(checkpoints_dir.glob("*.pth"))
            if checkpoints:
                print(f"   Checkpoints: {len(checkpoints)} arquivo(s)")
                for ckpt in checkpoints[:3]:  # Mostrar primeiros 3
                    print(f"      • {ckpt.name}")
                if len(checkpoints) > 3:
                    print(f"      ... e mais {len(checkpoints) - 3}")

            found_models.append(key)
        else:
            print(f"❌ {info['name']}: NÃO ENCONTRADO")

        print()

    if not found_models:
        print("💡 Nenhum modelo encontrado localmente.")
        print("   Use --download para baixar modelos pré-treinados.")

    return found_models


def main():
    parser = argparse.ArgumentParser(
        description="Baixar modelos pré-treinados do Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Listar modelos disponíveis
  python download_models.py --list
  
  # Verificar modelos já baixados
  python download_models.py --check
  
  # Baixar todos os modelos
  python download_models.py --all
  
  # Baixar modelo específico
  python download_models.py --model mnist
  python download_models.py --model cifar10
  python download_models.py --model fashion-mnist

Após baixar os modelos, você pode gerar imagens com:
  
  ./run.sh
  → Opção 4 (Gerar por classe)
  → Escolher checkpoint baixado
        """,
    )

    parser.add_argument(
        "--list", action="store_true", help="Listar todos os modelos disponíveis"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verificar quais modelos já estão baixados localmente",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        help="Baixar um modelo específico",
    )
    parser.add_argument(
        "--all", action="store_true", help="Baixar todos os modelos disponíveis"
    )

    args = parser.parse_args()

    # Se nenhum argumento, mostrar ajuda
    if not any([args.list, args.check, args.model, args.all]):
        parser.print_help()
        sys.exit(0)

    # Listar modelos
    if args.list:
        list_models()
        sys.exit(0)

    # Verificar modelos existentes
    if args.check:
        check_existing_models()
        sys.exit(0)

    # Baixar modelo específico
    if args.model:
        success = download_model(args.model)
        sys.exit(0 if success else 1)

    # Baixar todos os modelos
    if args.all:
        print("\n" + "=" * 70)
        print("📦 BAIXANDO TODOS OS MODELOS")
        print("=" * 70)

        success_count = 0
        failed_models = []

        for model_key in MODELS.keys():
            if download_model(model_key):
                success_count += 1
            else:
                failed_models.append(model_key)

        print("\n" + "=" * 70)
        print("📊 RESUMO DO DOWNLOAD")
        print("=" * 70)
        print(f"   Sucessos: {success_count}/{len(MODELS)}")

        if failed_models:
            print(f"   Falhas: {', '.join(failed_models)}")
        else:
            print("   ✅ Todos os modelos baixados com sucesso!")

        print()
        sys.exit(0 if success_count == len(MODELS) else 1)


if __name__ == "__main__":
    main()
