#!/usr/bin/env python3
"""
Funções utilitárias para treinamento e visualização
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

# ====================================================================================
# Funções de salvamento e carregamento
# ====================================================================================


def save_checkpoint(
    generator, discriminator, optimizerG, optimizerD, epoch, losses, config, output_dir
):
    """
    Salva checkpoint completo do treinamento

    Args:
        generator: modelo do gerador
        discriminator: modelo do discriminador/critic
        optimizerG: otimizador do gerador
        optimizerD: otimizador do discriminador
        epoch: época atual
        losses: dicionário com histórico de perdas
        config: configurações do treinamento
        output_dir: diretório de saída
    """
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizerG_state_dict": optimizerG.state_dict(),
        "optimizerD_state_dict": optimizerD.state_dict(),
        "losses": losses,
        "config": config,
    }

    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)

    # Também salvar como "latest" para fácil retomada
    latest_path = os.path.join(output_dir, "checkpoint_latest.pth")
    torch.save(checkpoint, latest_path)

    print(f"💾 Checkpoint salvo: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path,
    generator,
    discriminator,
    optimizerG=None,
    optimizerD=None,
    device="cpu",
):
    """
    Carrega checkpoint do treinamento

    Args:
        checkpoint_path: caminho para o checkpoint
        generator: modelo do gerador
        discriminator: modelo do discriminador
        optimizerG: otimizador do gerador (opcional)
        optimizerD: otimizador do discriminador (opcional)
        device: dispositivo para carregar os modelos

    Returns:
        (epoch, losses, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    if optimizerG is not None:
        optimizerG.load_state_dict(checkpoint["optimizerG_state_dict"])

    if optimizerD is not None:
        optimizerD.load_state_dict(checkpoint["optimizerD_state_dict"])

    epoch = checkpoint["epoch"]
    losses = checkpoint.get("losses", {"G": [], "D": []})
    config = checkpoint.get("config", {})

    print(f"✅ Checkpoint carregado: {checkpoint_path} (época {epoch})")

    return epoch, losses, config


def save_config(config, output_dir):
    """Salva configuração em JSON"""
    config_path = os.path.join(output_dir, "config.json")

    # Adicionar timestamp
    config["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"📝 Configuração salva: {config_path}")


# ====================================================================================
# Funções de visualização
# ====================================================================================


def save_image_grid(images, output_path, nrow=8, normalize=True):
    """
    Salva grid de imagens

    Args:
        images: tensor de imagens [B, C, H, W]
        output_path: caminho para salvar
        nrow: número de imagens por linha
        normalize: se True, normaliza de [-1, 1] para [0, 1]
    """
    vutils.save_image(
        images,
        output_path,
        nrow=nrow,
        normalize=normalize,
        value_range=(-1, 1) if normalize else None,
    )


def plot_losses(losses, output_dir):
    """
    Plota gráfico de perdas do gerador e discriminador

    Args:
        losses: dicionário com listas de perdas {'G': [...], 'D': [...]}
        output_dir: diretório para salvar o gráfico
    """
    plt.figure(figsize=(10, 5))
    plt.title("Perdas do Gerador e Discriminador durante Treinamento")
    plt.plot(losses["G"], label="Gerador", alpha=0.7)
    plt.plot(losses["D"], label="Discriminador", alpha=0.7)
    plt.xlabel("Iterações")
    plt.ylabel("Perda")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "training_losses.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Gráfico de perdas salvo: {plot_path}")


def generate_samples(generator, num_samples, nz, device, output_path, nrow=8):
    """
    Gera amostras do gerador e salva como grid

    Args:
        generator: modelo do gerador
        num_samples: número de amostras a gerar
        nz: tamanho do vetor de ruído
        device: dispositivo (CPU/GPU)
        output_path: caminho para salvar a imagem
        nrow: número de imagens por linha
    """
    with torch.no_grad():
        noise = torch.randn(num_samples, nz, 1, 1, device=device)
        fake_images = generator(noise).detach().cpu()
        save_image_grid(fake_images, output_path, nrow=nrow)


# ====================================================================================
# Funções de logging
# ====================================================================================


class TrainingLogger:
    """Logger para acompanhar progresso do treinamento"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "training.log")

        # Criar arquivo de log
        with open(self.log_file, "w") as f:
            f.write(f"Treinamento iniciado em: {datetime.now()}\n")
            f.write("=" * 70 + "\n\n")

    def log(self, message, print_console=True):
        """Registra mensagem no log e opcionalmente imprime"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        # Escrever no arquivo
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")

        # Imprimir no console
        if print_console:
            print(log_message)

    def log_epoch(self, epoch, total_epochs, loss_G, loss_D, elapsed_time):
        """Registra informações de uma época"""
        message = (
            f"Época [{epoch}/{total_epochs}] | "
            f"Loss_G: {loss_G:.4f} | Loss_D: {loss_D:.4f} | "
            f"Tempo: {elapsed_time:.2f}s"
        )
        self.log(message)


# ====================================================================================
# Funções de utilidade
# ====================================================================================


def create_output_dir(base_dir, dataset_name, model_type):
    """
    Cria diretório de saída organizado

    Args:
        base_dir: diretório base (ex: 'outputs')
        dataset_name: nome do dataset
        model_type: tipo do modelo

    Returns:
        caminho do diretório criado
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, dataset_name, f"{model_type}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    return output_dir


def get_device(ngpu=1):
    """
    Detecta e retorna o melhor dispositivo disponível

    Args:
        ngpu: número de GPUs solicitadas

    Returns:
        (device, ngpu_actual)
    """
    if torch.cuda.is_available() and ngpu > 0:
        device = torch.device("cuda:0")
        ngpu_actual = min(ngpu, torch.cuda.device_count())
        print(f"✓ GPU detectada! Usando: {torch.cuda.get_device_name(0)}")
        print(
            f"  Memória total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        ngpu_actual = 0
        print("⚠ GPU não detectada, usando CPU (será mais lento)")

    return device, ngpu_actual


def print_model_summary(generator, discriminator):
    """Imprime resumo dos modelos"""
    from models import count_parameters

    print("\n" + "=" * 70)
    print("RESUMO DOS MODELOS")
    print("=" * 70)

    print(f"\n🎨 Gerador:")
    print(f"   Parâmetros treináveis: {count_parameters(generator):,}")

    print(f"\n🔍 Discriminador:")
    print(f"   Parâmetros treináveis: {count_parameters(discriminator):,}")

    print("\n" + "=" * 70)


def format_time(seconds):
    """Formata tempo em segundos para formato legível"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def estimate_remaining_time(elapsed, current_epoch, total_epochs):
    """Estima tempo restante de treinamento"""
    if current_epoch == 0:
        return "Calculando..."

    avg_time_per_epoch = elapsed / current_epoch
    remaining_epochs = total_epochs - current_epoch
    remaining_seconds = avg_time_per_epoch * remaining_epochs

    return format_time(remaining_seconds)
