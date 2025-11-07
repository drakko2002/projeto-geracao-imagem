#!/usr/bin/env python3
"""
Script unificado para treinamento de GANs
Suporta múltiplos datasets e arquiteturas

Uso:
    python train.py --dataset cifar10 --model dcgan --epochs 50
    python train.py --dataset fashion-mnist --model wgan-gp --epochs 100
    python train.py --list-datasets  # Lista datasets disponíveis
    python train.py --list-models    # Lista modelos disponíveis
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

from config import (
    get_dataset,
    get_dataset_info,
    get_model_config,
    list_available_datasets,
    list_available_models,
)

# Importar módulos do projeto
from models import get_model
from utils import (
    TrainingLogger,
    create_output_dir,
    estimate_remaining_time,
    format_time,
    generate_samples,
    get_device,
    plot_losses,
    print_model_summary,
    save_checkpoint,
    save_config,
)

# ====================================================================================
# Funções de treinamento
# ====================================================================================


def train_dcgan(generator, discriminator, dataloader, device, config, output_dir):
    """
    Treinamento DCGAN padrão

    Args:
        generator: modelo do gerador
        discriminator: modelo do discriminador
        dataloader: dataloader do dataset
        device: dispositivo (CPU/GPU)
        config: configurações de treinamento
        output_dir: diretório de saída
    """

    # Configurações
    epochs = config["epochs"]
    lr = config["lr"]
    beta1 = config["beta1"]
    nz = config["nz"]

    # Critério e otimizadores
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Labels reais e fake
    real_label = 1.0
    fake_label = 0.0

    # Ruído fixo para visualização
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Histórico de perdas
    losses = {"G": [], "D": []}

    # Logger
    logger = TrainingLogger(output_dir)
    logger.log(f"Iniciando treinamento DCGAN")
    logger.log(
        f"Dataset: {config['dataset']} | Épocas: {epochs} | Batch size: {config['batch_size']}"
    )

    print("\n" + "=" * 70)
    print("INICIANDO TREINAMENTO")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        for i, data in enumerate(dataloader):
            ############################
            # (1) Atualizar Discriminador: maximizar log(D(x)) + log(1 - D(G(z)))
            ############################
            discriminator.zero_grad()

            # Treinar com batch real
            real_data = data[0].to(device)
            batch_size = real_data.size(0)
            label = torch.full(
                (batch_size,), real_label, dtype=torch.float, device=device
            )

            output = discriminator(real_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Treinar com batch fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)

            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Atualizar Gerador: maximizar log(D(G(z)))
            ############################
            generator.zero_grad()
            label.fill_(real_label)

            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            # Salvar perdas
            losses["G"].append(errG.item())
            losses["D"].append(errD.item())

            # Log de progresso
            if i % 50 == 0:
                print(
                    f"[{epoch}/{epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
                )

        # Fim da época
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time

        logger.log_epoch(epoch + 1, epochs, errG.item(), errD.item(), epoch_time)

        # Gerar amostras
        if (epoch + 1) % 5 == 0 or epoch == 0:
            sample_path = os.path.join(output_dir, "samples", f"epoch_{epoch+1}.png")
            generate_samples(generator, 64, nz, device, sample_path)
            print(f"✓ Amostras salvas: {sample_path}")

        # Salvar checkpoint
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            save_checkpoint(
                generator,
                discriminator,
                optimizerG,
                optimizerD,
                epoch + 1,
                losses,
                config,
                checkpoint_dir,
            )

        # Estimativa de tempo restante
        remaining = estimate_remaining_time(elapsed_total, epoch + 1, epochs)
        print(f"⏱️  Tempo restante estimado: {remaining}\n")

    # Treinamento concluído
    total_time = time.time() - start_time
    logger.log(f"\n✅ Treinamento concluído em {format_time(total_time)}")

    # Plotar perdas
    plot_losses(losses, output_dir)

    # Gerar amostras finais
    final_sample_path = os.path.join(output_dir, "final_samples.png")
    generate_samples(generator, 64, nz, device, final_sample_path)

    print("\n" + "=" * 70)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 70)
    print(f"📁 Resultados salvos em: {output_dir}")
    print(f"⏱️  Tempo total: {format_time(total_time)}")
    print("=" * 70 + "\n")


def compute_gradient_penalty(critic, real_data, fake_data, device):
    """Calcula gradient penalty para WGAN-GP"""
    batch_size = real_data.size(0)

    # Gerar alpha aleatório
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    # Interpolar entre dados reais e fake
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    # Calcular output do critic
    d_interpolates = critic(interpolates)

    # Calcular gradientes
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)

    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def train_wgan_gp(generator, critic, dataloader, device, config, output_dir):
    """
    Treinamento WGAN-GP (Wasserstein GAN with Gradient Penalty)

    Args:
        generator: modelo do gerador
        critic: modelo do critic
        dataloader: dataloader do dataset
        device: dispositivo (CPU/GPU)
        config: configurações de treinamento
        output_dir: diretório de saída
    """

    # Configurações
    epochs = config["epochs"]
    lr = config["lr"]
    beta1 = config["beta1"]
    nz = config["nz"]
    n_critic = config.get("n_critic", 5)
    lambda_gp = config.get("lambda_gp", 10.0)

    # Otimizadores
    optimizerD = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Ruído fixo para visualização
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Histórico de perdas
    losses = {"G": [], "D": []}

    # Logger
    logger = TrainingLogger(output_dir)
    logger.log(f"Iniciando treinamento WGAN-GP")
    logger.log(
        f"Dataset: {config['dataset']} | Épocas: {epochs} | Batch size: {config['batch_size']}"
    )
    logger.log(f"n_critic: {n_critic} | lambda_gp: {lambda_gp}")

    print("\n" + "=" * 70)
    print("INICIANDO TREINAMENTO WGAN-GP")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        for i, data in enumerate(dataloader):
            real_data = data[0].to(device)
            batch_size = real_data.size(0)

            ############################
            # (1) Atualizar Critic: minimizar -E[D(x)] + E[D(G(z))] + lambda * GP
            ############################
            for _ in range(n_critic):
                critic.zero_grad()

                # Critic score para dados reais
                critic_real = critic(real_data).mean()

                # Critic score para dados fake
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = generator(noise)
                critic_fake = critic(fake.detach()).mean()

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(
                    critic, real_data, fake, device
                )

                # Perda do critic (Wasserstein distance + gradient penalty)
                errD = -critic_real + critic_fake + lambda_gp * gradient_penalty
                errD.backward()
                optimizerD.step()

            ############################
            # (2) Atualizar Gerador: minimizar -E[D(G(z))]
            ############################
            generator.zero_grad()

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)
            critic_fake = critic(fake).mean()

            errG = -critic_fake
            errG.backward()
            optimizerG.step()

            # Salvar perdas
            losses["G"].append(errG.item())
            losses["D"].append(errD.item())

            # Log de progresso
            if i % 50 == 0:
                print(
                    f"[{epoch}/{epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(x): {critic_real.item():.4f} D(G(z)): {critic_fake.item():.4f}"
                )

        # Fim da época
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time

        logger.log_epoch(epoch + 1, epochs, errG.item(), errD.item(), epoch_time)

        # Gerar amostras
        if (epoch + 1) % 5 == 0 or epoch == 0:
            sample_path = os.path.join(output_dir, "samples", f"epoch_{epoch+1}.png")
            generate_samples(generator, 64, nz, device, sample_path)
            print(f"✓ Amostras salvas: {sample_path}")

        # Salvar checkpoint
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            save_checkpoint(
                generator,
                critic,
                optimizerG,
                optimizerD,
                epoch + 1,
                losses,
                config,
                checkpoint_dir,
            )

        # Estimativa de tempo restante
        remaining = estimate_remaining_time(elapsed_total, epoch + 1, epochs)
        print(f"⏱️  Tempo restante estimado: {remaining}\n")

    # Treinamento concluído
    total_time = time.time() - start_time
    logger.log(f"\n✅ Treinamento concluído em {format_time(total_time)}")

    # Plotar perdas
    plot_losses(losses, output_dir)

    # Gerar amostras finais
    final_sample_path = os.path.join(output_dir, "final_samples.png")
    generate_samples(generator, 64, nz, device, final_sample_path)

    print("\n" + "=" * 70)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 70)
    print(f"📁 Resultados salvos em: {output_dir}")
    print(f"⏱️  Tempo total: {format_time(total_time)}")
    print("=" * 70 + "\n")


# ====================================================================================
# Main
# ====================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Script unificado para treinamento de GANs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python train.py --dataset cifar10 --model dcgan --epochs 50
  python train.py --dataset fashion-mnist --model wgan-gp --epochs 100 --batch-size 64
  python train.py --dataset mnist --model dcgan --img-size 32 --epochs 25
  python train.py --list-datasets
  python train.py --list-models
        """,
    )

    # Argumentos principais
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "mnist", "fashion-mnist", "celeba", "custom"],
        help="Dataset para treinamento",
    )
    parser.add_argument(
        "--model", type=str, choices=["dcgan", "wgan-gp"], help="Tipo de modelo GAN"
    )

    # Configurações de treinamento
    parser.add_argument(
        "--epochs", type=int, default=50, help="Número de épocas (padrão: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Tamanho do batch (padrão: 128)"
    )
    parser.add_argument(
        "--img-size", type=int, default=64, help="Tamanho das imagens (padrão: 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (usa padrão do modelo se não especificado)",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=None,
        help="Beta1 do Adam (usa padrão do modelo se não especificado)",
    )

    # Configurações de modelo
    parser.add_argument(
        "--nz",
        type=int,
        default=100,
        help="Tamanho do vetor de ruído latente (padrão: 100)",
    )
    parser.add_argument(
        "--ngf", type=int, default=64, help="Número de filtros do gerador (padrão: 64)"
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=64,
        help="Número de filtros do discriminador (padrão: 64)",
    )

    # Configurações de sistema
    parser.add_argument(
        "--dataroot",
        type=str,
        default="./data",
        help="Diretório raiz dos datasets (padrão: ./data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Diretório de saída (padrão: ./outputs)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Número de workers para DataLoader (padrão: 2)",
    )
    parser.add_argument(
        "--ngpu", type=int, default=1, help="Número de GPUs (padrão: 1)"
    )

    # Resumir modelos/datasets
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Caminho para checkpoint para continuar treinamento",
    )

    # Utilitários
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Lista todos os datasets disponíveis e sai",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Lista todos os modelos disponíveis e sai",
    )

    args = parser.parse_args()

    # Listar datasets/modelos e sair
    if args.list_datasets:
        list_available_datasets()
        return

    if args.list_models:
        list_available_models()
        return

    # Validar argumentos
    if args.dataset is None or args.model is None:
        parser.error(
            "--dataset e --model são obrigatórios. "
            "Use --list-datasets e --list-models para ver opções."
        )

    # Obter dispositivo
    device, ngpu = get_device(args.ngpu)

    # Obter configuração do modelo
    model_config = get_model_config(args.model)

    # Atualizar configurações com argumentos CLI
    if args.lr is None:
        args.lr = model_config["default_lr"]
    if args.beta1 is None:
        args.beta1 = model_config["default_beta1"]

    # Carregar dataset
    print("\n📦 Carregando dataset...")
    dataloader, nc = get_dataset(
        args.dataset,
        dataroot=args.dataroot,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(f"✓ Dataset carregado: {len(dataloader.dataset)} imagens")

    # Criar modelos
    print("\n🤖 Criando modelos...")
    model_cfg = {
        "nz": args.nz,
        "ngf": args.ngf,
        "ndf": args.ndf,
        "nc": nc,
        "img_size": args.img_size,
    }

    generator, discriminator = get_model(args.model, model_cfg)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Resumo dos modelos
    print_model_summary(generator, discriminator)

    # Criar diretório de saída
    output_dir = create_output_dir(args.output, args.dataset, args.model)
    print(f"\n📁 Diretório de saída: {output_dir}")

    # Salvar configuração
    config = {
        "dataset": args.dataset,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "lr": args.lr,
        "beta1": args.beta1,
        "nz": args.nz,
        "ngf": args.ngf,
        "ndf": args.ndf,
        "nc": nc,
        "ngpu": ngpu,
    }

    # Adicionar configurações específicas do modelo
    if args.model == "wgan-gp":
        config["n_critic"] = model_config.get("n_critic", 5)
        config["lambda_gp"] = model_config.get("lambda_gp", 10.0)

    save_config(config, output_dir)

    # Treinar
    if args.model == "dcgan":
        train_dcgan(generator, discriminator, dataloader, device, config, output_dir)
    elif args.model == "wgan-gp":
        train_wgan_gp(generator, discriminator, dataloader, device, config, output_dir)


if __name__ == "__main__":
    main()
