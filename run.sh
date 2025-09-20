#!/bin/bash

# Script para rodar o projeto de geração de imagens

# Ativar ambiente virtual
source venv/bin/activate

# Carregar variáveis do arquivo .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Erro: Arquivo .env não encontrado!"
    echo "Copie .env.example para .env e configure seu token do HuggingFace"
    exit 1
fi

# Executar aplicação
python app.py

echo "Imagem gerada! Verifique o arquivo 'saida.png'"