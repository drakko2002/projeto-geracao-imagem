#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Script para preparar modelos treinados para upload
# Empacota checkpoints e remove arquivos desnecessários
# ══════════════════════════════════════════════════════════════

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║      PREPARAR MODELOS PARA TRANSFERÊNCIA (GOOGLE DRIVE)     ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Criar diretório para modelos empacotados
PRETRAINED_DIR="pretrained_models"
mkdir -p "$PRETRAINED_DIR"

echo "📦 Procurando modelos treinados..."
echo ""

MODELS_FOUND=0

# ══════════════════════════════════════════════════════════════
# MNIST
# ══════════════════════════════════════════════════════════════

if [ -d "outputs/mnist" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 MNIST"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd outputs/mnist
    LATEST_DIR=$(ls -td dcgan_* 2>/dev/null | head -1)
    
    if [ -n "$LATEST_DIR" ] && [ -d "$LATEST_DIR/checkpoints" ]; then
        echo "✅ Encontrado: $LATEST_DIR"
        
        # Contar checkpoints
        CHECKPOINT_COUNT=$(ls "$LATEST_DIR/checkpoints"/*.pth 2>/dev/null | wc -l)
        echo "   Checkpoints: $CHECKPOINT_COUNT arquivo(s)"
        
        # Criar arquivo zip com apenas checkpoint final e config
        echo "   📦 Empacotando..."
        
        # Criar estrutura temporária
        TEMP_DIR="temp_mnist"
        mkdir -p "$TEMP_DIR/checkpoints"
        
        # Copiar apenas checkpoint_latest.pth e config.json
        if [ -f "$LATEST_DIR/checkpoints/checkpoint_latest.pth" ]; then
            cp "$LATEST_DIR/checkpoints/checkpoint_latest.pth" "$TEMP_DIR/checkpoints/"
        fi
        
        if [ -f "$LATEST_DIR/config.json" ]; then
            cp "$LATEST_DIR/config.json" "$TEMP_DIR/"
        fi
        
        # Criar README dentro do zip
        cat > "$TEMP_DIR/README.txt" << EOF
MNIST DCGAN - Modelo Pré-Treinado
═══════════════════════════════════════════════════════════

Dataset: MNIST (dígitos 0-9)
Modelo: DCGAN
Resolução: 28x28 pixels (grayscale)
Épocas treinadas: $(grep -o '"epoch": [0-9]*' "$LATEST_DIR/config.json" | tail -1 | grep -o '[0-9]*' || echo "?")

Uso:
  python generate_interactive.py \\
    --checkpoint outputs/mnist/dcgan_pretrained/checkpoints/checkpoint_latest.pth \\
    --prompt "número 5"

Classes disponíveis: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
EOF
        
        # Criar zip
        cd "$TEMP_DIR"
        zip -r "../../../$PRETRAINED_DIR/mnist_checkpoint.zip" . -q
        cd ..
        
        # Remover temporário
        rm -rf "$TEMP_DIR"
        
        SIZE=$(du -h "../../$PRETRAINED_DIR/mnist_checkpoint.zip" | cut -f1)
        echo "   ✅ Salvo: $PRETRAINED_DIR/mnist_checkpoint.zip ($SIZE)"
        
        MODELS_FOUND=$((MODELS_FOUND + 1))
    else
        echo "❌ Nenhum checkpoint encontrado"
    fi
    
    cd ../..
    echo ""
fi

# ══════════════════════════════════════════════════════════════
# CIFAR-10
# ══════════════════════════════════════════════════════════════

if [ -d "outputs/cifar10" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 CIFAR-10"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd outputs/cifar10
    LATEST_DIR=$(ls -td dcgan_* 2>/dev/null | head -1)
    
    if [ -n "$LATEST_DIR" ] && [ -d "$LATEST_DIR/checkpoints" ]; then
        echo "✅ Encontrado: $LATEST_DIR"
        
        CHECKPOINT_COUNT=$(ls "$LATEST_DIR/checkpoints"/*.pth 2>/dev/null | wc -l)
        echo "   Checkpoints: $CHECKPOINT_COUNT arquivo(s)"
        
        echo "   📦 Empacotando..."
        
        TEMP_DIR="temp_cifar10"
        mkdir -p "$TEMP_DIR/checkpoints"
        
        if [ -f "$LATEST_DIR/checkpoints/checkpoint_latest.pth" ]; then
            cp "$LATEST_DIR/checkpoints/checkpoint_latest.pth" "$TEMP_DIR/checkpoints/"
        fi
        
        if [ -f "$LATEST_DIR/config.json" ]; then
            cp "$LATEST_DIR/config.json" "$TEMP_DIR/"
        fi
        
        cat > "$TEMP_DIR/README.txt" << EOF
CIFAR-10 DCGAN - Modelo Pré-Treinado
═══════════════════════════════════════════════════════════

Dataset: CIFAR-10
Modelo: DCGAN
Resolução: 32x32 pixels (RGB colorido)
Épocas treinadas: $(grep -o '"epoch": [0-9]*' "$LATEST_DIR/config.json" | tail -1 | grep -o '[0-9]*' || echo "?")

Uso:
  python generate_interactive.py \\
    --checkpoint outputs/cifar10/dcgan_pretrained/checkpoints/checkpoint_latest.pth \\
    --prompt "gerar um gato"

Classes disponíveis:
  Animais: Pássaros, Gatos, Cervos, Cachorros, Sapos, Cavalos
  Veículos: Aviões, Carros, Navios, Caminhões
EOF
        
        cd "$TEMP_DIR"
        zip -r "../../../$PRETRAINED_DIR/cifar10_checkpoint.zip" . -q
        cd ..
        rm -rf "$TEMP_DIR"
        
        SIZE=$(du -h "../../$PRETRAINED_DIR/cifar10_checkpoint.zip" | cut -f1)
        echo "   ✅ Salvo: $PRETRAINED_DIR/cifar10_checkpoint.zip ($SIZE)"
        
        MODELS_FOUND=$((MODELS_FOUND + 1))
    else
        echo "❌ Nenhum checkpoint encontrado"
    fi
    
    cd ../..
    echo ""
fi

# ══════════════════════════════════════════════════════════════
# Fashion-MNIST
# ══════════════════════════════════════════════════════════════

if [ -d "outputs/fashion-mnist" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 Fashion-MNIST"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd outputs/fashion-mnist
    LATEST_DIR=$(ls -td dcgan_* 2>/dev/null | head -1)
    
    if [ -n "$LATEST_DIR" ] && [ -d "$LATEST_DIR/checkpoints" ]; then
        echo "✅ Encontrado: $LATEST_DIR"
        
        CHECKPOINT_COUNT=$(ls "$LATEST_DIR/checkpoints"/*.pth 2>/dev/null | wc -l)
        echo "   Checkpoints: $CHECKPOINT_COUNT arquivo(s)"
        
        echo "   📦 Empacotando..."
        
        TEMP_DIR="temp_fashion"
        mkdir -p "$TEMP_DIR/checkpoints"
        
        if [ -f "$LATEST_DIR/checkpoints/checkpoint_latest.pth" ]; then
            cp "$LATEST_DIR/checkpoints/checkpoint_latest.pth" "$TEMP_DIR/checkpoints/"
        fi
        
        if [ -f "$LATEST_DIR/config.json" ]; then
            cp "$LATEST_DIR/config.json" "$TEMP_DIR/"
        fi
        
        cat > "$TEMP_DIR/README.txt" << EOF
Fashion-MNIST DCGAN - Modelo Pré-Treinado
═══════════════════════════════════════════════════════════

Dataset: Fashion-MNIST (roupas e acessórios)
Modelo: DCGAN
Resolução: 28x28 pixels (grayscale)
Épocas treinadas: $(grep -o '"epoch": [0-9]*' "$LATEST_DIR/config.json" | tail -1 | grep -o '[0-9]*' || echo "?")

Uso:
  python generate_interactive.py \\
    --checkpoint outputs/fashion-mnist/dcgan_pretrained/checkpoints/checkpoint_latest.pth \\
    --prompt "camiseta"

Classes disponíveis:
  Camiseta, Calça, Pullover, Vestido, Casaco
  Sandália, Camisa, Tênis, Bolsa, Bota
EOF
        
        cd "$TEMP_DIR"
        zip -r "../../../$PRETRAINED_DIR/fashion-mnist_checkpoint.zip" . -q
        cd ..
        rm -rf "$TEMP_DIR"
        
        SIZE=$(du -h "../../$PRETRAINED_DIR/fashion-mnist_checkpoint.zip" | cut -f1)
        echo "   ✅ Salvo: $PRETRAINED_DIR/fashion-mnist_checkpoint.zip ($SIZE)"
        
        MODELS_FOUND=$((MODELS_FOUND + 1))
    else
        echo "❌ Nenhum checkpoint encontrado"
    fi
    
    cd ../..
    echo ""
fi

# ══════════════════════════════════════════════════════════════
# RESUMO
# ══════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 RESUMO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $MODELS_FOUND -eq 0 ]; then
    echo "❌ Nenhum modelo encontrado para empacotar"
    echo ""
    echo "💡 Treine modelos primeiro:"
    echo "   ./run.sh → Opção 2 (Exemplos rápidos)"
    echo ""
else
    echo "✅ $MODELS_FOUND modelo(s) empacotado(s) com sucesso!"
    echo ""
    echo "📁 Arquivos criados em: $PRETRAINED_DIR/"
    echo ""
    ls -lh "$PRETRAINED_DIR"/*.zip 2>/dev/null || true
    echo ""
    
    TOTAL_SIZE=$(du -sh "$PRETRAINED_DIR" | cut -f1)
    echo "💾 Tamanho total: $TOTAL_SIZE"
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📤 PRÓXIMOS PASSOS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. Faça upload dos arquivos .zip para o Google Drive"
    echo "   → Acesse: https://drive.google.com"
    echo "   → Crie pasta: 'GAN_Pretrained_Models'"
    echo "   → Upload dos arquivos de: $PRETRAINED_DIR/"
    echo ""
    echo "2. Compartilhe cada arquivo:"
    echo "   → Clique direito → Compartilhar"
    echo "   → 'Qualquer pessoa com o link'"
    echo "   → Permissão: 'Leitor'"
    echo "   → Copiar link"
    echo ""
    echo "3. Extrair IDs dos links:"
    echo "   Link: https://drive.google.com/file/d/SEU_ID_AQUI/view"
    echo "                                         ^^^^^^^^^^^^"
    echo "   Copie apenas a parte do ID"
    echo ""
    echo "4. Configurar IDs em download_models.py:"
    echo "   Edite as linhas com 'google_drive_id' e substitua 'SEU_ID_AQUI'"
    echo ""
    echo "5. Commit e push:"
    echo "   git add download_models.py"
    echo "   git commit -m 'Add Google Drive IDs for pretrained models'"
    echo "   git push origin main"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📖 Guia completo: TRANSFER_GUIDE.md"
    echo ""
fi

echo "✨ Pronto!"
echo ""
