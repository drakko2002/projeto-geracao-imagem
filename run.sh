#!/bin/bash

# Script para rodar o projeto de geração de imagens

echo "🚀 Iniciando IA Image Generator..."

# Verificar se Python está disponível
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado! Instale Python 3.10+ primeiro."
    exit 1
fi

echo "✅ Python encontrado: $(python3 --version)"

# Criar ambiente virtual se não existir
if [ ! -d "venv" ]; then
    echo "📦 Criando ambiente virtual..."
    python3 -m venv venv
fi

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

# Verificar se dependências estão instaladas
if [ ! -f "venv/lib/python*/site-packages/torch*" ]; then
    echo "📥 Instalando dependências (primeira execução)..."
    echo "   Instalando PyTorch com CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo "   Instalando outras dependências..."
    pip install -r requirements.txt
fi

# Carregar variáveis do arquivo .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ Arquivo .env não encontrado!"
    echo "   Copie .env.example para .env e configure seu token do HuggingFace"
    echo "   Comando: cp .env.example .env"
    echo "   Token: https://huggingface.co/settings/tokens"
    exit 1
fi

# Verificar se modelo existe
if [ ! -d "model" ]; then
    echo "🔽 Baixando modelo Stable Diffusion (primeira execução)..."
    python download_model.py
fi

# Executar aplicação
echo "🎨 Gerando imagem..."
python app.py

echo "✅ Imagem gerada! Verifique o arquivo 'saida.png'"