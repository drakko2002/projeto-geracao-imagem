#!/bin/bash
# Script de exemplo para começar rapidamente

echo "🎨 Exemplo de Treinamento Rápido com GANs"
echo "=========================================="
echo ""
echo "Escolha um exemplo:"
echo ""
echo "1) MNIST + DCGAN (5 épocas, ~5min)"
echo "2) Fashion-MNIST + DCGAN (25 épocas, ~15min)"
echo "3) CIFAR-10 + DCGAN (50 épocas, ~1h)"
echo "4) CIFAR-10 + WGAN-GP (100 épocas, ~3h)"
echo "5) Gerar imagens de modelo existente"
echo "6) Listar datasets disponíveis"
echo "7) Listar modelos disponíveis"
echo ""
read -p "Escolha (1-7): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Treinando DCGAN com MNIST (teste rápido)..."
        python train.py --dataset mnist --model dcgan --epochs 5 --batch-size 128
        ;;
    2)
        echo ""
        echo "🚀 Treinando DCGAN com Fashion-MNIST..."
        python train.py --dataset fashion-mnist --model dcgan --epochs 25 --batch-size 128
        ;;
    3)
        echo ""
        echo "🚀 Treinando DCGAN com CIFAR-10..."
        python train.py --dataset cifar10 --model dcgan --epochs 50 --batch-size 128
        ;;
    4)
        echo ""
        echo "🚀 Treinando WGAN-GP com CIFAR-10 (alta qualidade)..."
        python train.py --dataset cifar10 --model wgan-gp --epochs 100 --batch-size 64
        ;;
    5)
        echo ""
        echo "📁 Procurando checkpoints..."
        find outputs -name "checkpoint_latest.pth" -type f 2>/dev/null
        echo ""
        read -p "Cole o caminho do checkpoint: " checkpoint
        python generate.py --checkpoint "$checkpoint" --num-samples 64
        ;;
    6)
        python train.py --list-datasets
        ;;
    7)
        python train.py --list-models
        ;;
    *)
        echo "❌ Opção inválida!"
        exit 1
        ;;
esac

echo ""
echo "✅ Concluído!"
