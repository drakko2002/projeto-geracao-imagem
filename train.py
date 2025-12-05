#!/usr/bin/env python3
"""
Script unificado para treinamento de GANs
Suporta múltiplos datasets e arquiteturas

Uso:
    python train.py --dataset cifar10 --model dcgan --epochs 50
    python train.py --dataset fashion-mnist --model wgan-gp --epochs 100
    python train.py --dataset mnist --model dcgan-cond --epochs 50
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
import torchvision.utils as vutils

# ====================================================================================
# Constantes
# ====================================================================================

MIN_RESOLUTION = 128  # Resolução mínima suportada

from config import (
    get_dataset,
    get_dataset_info,
    get_model_config,
    list_available_datasets,
    list_available_models,
    DATASET_CONFIGS,   # << usado para saber num_classes por dataset
)

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
    Treinamento DCGAN padrão OU condicional (dcgan-cond)

    Se config["is_conditional"] = True:
        - Usa labels do dataset (data[1])
        - Usa gerador/discriminador condicionais (dcgan-cond)
        - Prompt / GUI podem mapear texto -> classe depois
    """

    # Configurações comuns
    epochs = config["epochs"]
    lr = config["lr"]
    beta1 = config["beta1"]
    nz = config["nz"]

    # Flags para condicional
    is_conditional = config.get("is_conditional", False)
    num_classes = config.get("num_classes", None)

    if is_conditional and num_classes is None:
        raise ValueError("Treino condicional requer 'num_classes' em config.")

    # Critério e otimizadores
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Se condicional, gera labels fixos p/ visualização (ciclo sobre classes)
    if is_conditional:
        fixed_labels = torch.arange(0, num_classes, device=device)
        # repete até 64
        fixed_labels = fixed_labels.repeat((64 + num_classes - 1) // num_classes)[:64]
    else:
        fixed_labels = None

    losses = {"G": [], "D": []}
    logger = TrainingLogger(output_dir)

    logger.log(
        f"Iniciando treinamento {'DCGAN Condicional' if is_conditional else 'DCGAN'}"
    )
    logger.log(
        f"Dataset: {config['dataset']} | Épocas: {epochs} | Batch size: {config['batch_size']}"
    )

    print("\n" + "=" * 70)
    print("INICIANDO TREINAMENTO", "CONDICIONAL" if is_conditional else "")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        for i, data in enumerate(dataloader):
            # ------------------------------------------------
            # Preparar batch real
            # ------------------------------------------------
            if is_conditional:
                real_data, labels = data[0].to(device), data[1].to(device)
            else:
                real_data = data[0].to(device)
                labels = None  # não usado

            batch_size = real_data.size(0)

            ############################
            # (1) Atualizar D
            ############################
            discriminator.zero_grad()

            # --- Real ---
            label_real_tensor = torch.full(
                (batch_size,), real_label, dtype=torch.float, device=device
            )

            if is_conditional:
                output_real = discriminator(real_data, labels).view(-1)
            else:
                output_real = discriminator(real_data).view(-1)

            errD_real = criterion(output_real, label_real_tensor)
            errD_real.backward()
            D_x = output_real.mean().item()

            # --- Fake ---
            noise = torch.randn(batch_size, nz, 1, 1, device=device)

            if is_conditional:
                # amostra rótulos aleatórios para fakes
                fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
                fake = generator(noise, fake_labels)
                output_fake = discriminator(fake.detach(), fake_labels).view(-1)
            else:
                fake = generator(noise)
                output_fake = discriminator(fake.detach()).view(-1)

            label_fake_tensor = torch.full(
                (batch_size,), fake_label, dtype=torch.float, device=device
            )
            errD_fake = criterion(output_fake, label_fake_tensor)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Atualizar G
            ############################
            generator.zero_grad()
            label_gen_tensor = torch.full(
                (batch_size,), real_label, dtype=torch.float, device=device
            )

            noise = torch.randn(batch_size, nz, 1, 1, device=device)

            if is_conditional:
                gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
                fake = generator(noise, gen_labels)
                output = discriminator(fake, gen_labels).view(-1)
            else:
                fake = generator(noise)
                output = discriminator(fake).view(-1)

            errG = criterion(output, label_gen_tensor)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            losses["G"].append(errG.item())
            losses["D"].append(errD.item())

            if i % 50 == 0:
                print(
                    f"[{epoch+1}/{epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
                )

        # ---------------- Fim da época ----------------
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time

        logger.log_epoch(epoch + 1, epochs, errG.item(), errD.item(), epoch_time)

        # Amostras
        if (epoch + 1) % 5 == 0 or epoch == 0:
            samples_dir = os.path.join(output_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            sample_path = os.path.join(samples_dir, f"epoch_{epoch+1}.png")

            if is_conditional and fixed_labels is not None:
                # Modelos condicionais: gera usando labels fixos (0..num_classes-1 repetidos)
                with torch.no_grad():
                    fake_samples = generator(fixed_noise, fixed_labels).detach().cpu()
                # saída do gerador está em [-1, 1] -> normaliza pra [0, 1]
                fake_samples = (fake_samples + 1) / 2
                vutils.save_image(
                    fake_samples,
                    sample_path,
                    nrow=8,
                )
            else:
                # Modelos não-condicionais usam o helper padrão
                generate_samples(generator, 64, nz, device, sample_path)

            print(f"✓ Amostras salvas: {sample_path}")


        # Checkpoints
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

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

        remaining = estimate_remaining_time(elapsed_total, epoch + 1, epochs)
        print(f"⏱️  Tempo restante estimado: {remaining}\n")

    total_time = time.time() - start_time
    logger.log(f"\n Treinamento concluído em {format_time(total_time)}")

    plot_losses(losses, output_dir)

    final_sample_path = os.path.join(output_dir, "final_samples.png")
    if is_conditional and fixed_labels is not None:
        with torch.no_grad():
            fake_samples = generator(fixed_noise, fixed_labels).detach().cpu()
        fake_samples = (fake_samples + 1) / 2  # [-1,1] -> [0,1]
        vutils.save_image(
            fake_samples,
            final_sample_path,
            nrow=8,
        )
    else:
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

    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    d_interpolates = critic(interpolates)

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
    Treinamento WGAN-GP (sem condicionamento)
    """

    epochs = config["epochs"]
    lr = config["lr"]
    beta1 = config["beta1"]
    nz = config["nz"]
    n_critic = config.get("n_critic", 5)
    lambda_gp = config.get("lambda_gp", 10.0)

    optimizerD = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    losses = {"G": [], "D": []}
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

            # (1) Atualizar Critic n_critic vezes
            for _ in range(n_critic):
                critic.zero_grad()

                critic_real = critic(real_data).mean()

                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = generator(noise)
                critic_fake = critic(fake.detach()).mean()

                gradient_penalty = compute_gradient_penalty(
                    critic, real_data, fake, device
                )

                errD = -critic_real + critic_fake + lambda_gp * gradient_penalty
                errD.backward()
                optimizerD.step()

            # (2) Atualizar Gerador
            generator.zero_grad()
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)
            critic_fake = critic(fake).mean()
            errG = -critic_fake
            errG.backward()
            optimizerG.step()

            losses["G"].append(errG.item())
            losses["D"].append(errD.item())

            if i % 50 == 0:
                print(
                    f"[{epoch+1}/{epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(x): {critic_real.item():.4f} D(G(z)): {critic_fake.item():.4f}"
                )

        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time

        logger.log_epoch(epoch + 1, epochs, errG.item(), errD.item(), epoch_time)

        # Amostras
        if (epoch + 1) % 5 == 0 or epoch == 0:
            samples_dir = os.path.join(output_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            sample_path = os.path.join(samples_dir, f"epoch_{epoch+1}.png")

            # WGAN-GP é não condicional, usa helper padrão
            generate_samples(generator, 64, nz, device, sample_path)

            print(f"✓ Amostras salvas: {sample_path}")

        # Checkpoints
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
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

        remaining = estimate_remaining_time(elapsed_total, epoch + 1, epochs)
        print(f"⏱️  Tempo restante estimado: {remaining}\n")

    total_time = time.time() - start_time
    logger.log(f"\n Treinamento concluído em {format_time(total_time)}")

    plot_losses(losses, output_dir)

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
  python train.py --dataset mnist --model dcgan-cond --epochs 25
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
        "--model",
        type=str,
        choices=["dcgan", "dcgan-cond", "wgan-gp"],
        help="Tipo de modelo GAN",
    )

    # Configurações de treinamento
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Tamanho da imagem (padrão: 128). Presets recomendados: 128, 256. Use potências de 2 (64/128/256).",
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
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument(
        "--ngf",
        type=int,
        default=64,
        help="Filtros do gerador (padrão: 64). Para 256px, recomenda-se 96-128 para melhor qualidade.",
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=64,
        help="Filtros do discriminador (padrão: 64). Para 256px, recomenda-se 96-128 para melhor qualidade.",
    )

    # Configurações de sistema
    parser.add_argument("--dataroot", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./outputs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ngpu", type=int, default=1)

    # Utilitários
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--list-models", action="store_true")

    args = parser.parse_args()

    # Listagens
    if args.list_datasets:
        list_available_datasets()
        return

    if args.list_models:
        list_available_models()
        return

    # Validação básica
    if args.dataset is None or args.model is None:
        parser.error(
            "--dataset e --model são obrigatórios. "
            "Use --list-datasets e --list-models para ver opções."
        )
    
    # Validar resolução mínima
    if args.img_size < MIN_RESOLUTION:
        parser.error(
            f"Resolução mínima é {MIN_RESOLUTION}px (fornecido: {args.img_size}). "
            f"Use --img-size {MIN_RESOLUTION} ou 256 para melhores resultados."
        )

    # Dispositivo
    device, ngpu = get_device(args.ngpu)

    # Para dcgan-cond, usamos config base do dcgan
    base_model_type = "dcgan" if args.model == "dcgan-cond" else args.model
    model_config_defaults = get_model_config(base_model_type)

    if args.lr is None:
        args.lr = model_config_defaults["default_lr"]
    if args.beta1 is None:
        args.beta1 = model_config_defaults["default_beta1"]

    # Dataset
    print("\n📦 Carregando dataset...")
    dataloader, nc = get_dataset(
        args.dataset,
        dataroot=args.dataroot,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(f"✓ Dataset carregado: {len(dataloader.dataset)} imagens")

    # Descobrir num_classes se dataset suportar
    num_classes = None
    if args.dataset in DATASET_CONFIGS:
        ds_cfg = DATASET_CONFIGS[args.dataset]
        classes = ds_cfg.get("classes", [])
        if classes:
            num_classes = len(classes)

    # Criar modelos
    print("\n🤖 Criando modelos...")
    model_cfg = {
        "nz": args.nz,
        "ngf": args.ngf,
        "ndf": args.ndf,
        "nc": nc,
        "img_size": args.img_size,
    }

    is_conditional = args.model == "dcgan-cond"
    if is_conditional:
        if num_classes is None:
            raise ValueError(
                f"O dataset '{args.dataset}' não possui 'classes' definidas em DATASET_CONFIGS, "
                "necessário para dcgan-cond."
            )
        model_cfg["num_classes"] = num_classes

    generator, discriminator_or_critic = get_model(args.model, model_cfg)
    generator = generator.to(device)
    discriminator_or_critic = discriminator_or_critic.to(device)

    print_model_summary(generator, discriminator_or_critic)

    # Diretório de saída
    output_dir = create_output_dir(args.output, args.dataset, args.model)
    print(f"\n📁 Diretório de saída: {output_dir}")

    # Config base
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

    # Flags específicas
    if is_conditional:
        config["is_conditional"] = True
        config["num_classes"] = num_classes
        config["text_conditional"] = True  # usado pelo app_gui para saber que entende prompt
    if args.model == "wgan-gp":
        config["n_critic"] = model_config_defaults.get("n_critic", 5)
        config["lambda_gp"] = model_config_defaults.get("lambda_gp", 10.0)

    save_config(config, output_dir)

    # Treino
    if args.model in ("dcgan", "dcgan-cond"):
        train_dcgan(generator, discriminator_or_critic, dataloader, device, config, output_dir)
    elif args.model == "wgan-gp":
        train_wgan_gp(generator, discriminator_or_critic, dataloader, device, config, output_dir)

if __name__ == "__main__":
    main()
