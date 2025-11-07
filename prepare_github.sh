#!/bin/bash
# Preparar projeto para push no GitHub

echo "🚀 Preparando projeto para GitHub..."
echo ""

# Verificar se estamos no diretório correto
if [ ! -f "train.py" ]; then
    echo "❌ Erro: Execute este script no diretório raiz do projeto"
    exit 1
fi

echo "1️⃣ Verificando estrutura do projeto..."
echo ""

# Arquivos essenciais
REQUIRED_FILES=(
    "train.py"
    "generate.py"
    "models.py"
    "config.py"
    "utils.py"
    "requirements.txt"
    "README.md"
    ".gitignore"
)

ALL_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (faltando!)"
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo "⚠️  Alguns arquivos essenciais estão faltando!"
    exit 1
fi

echo ""
echo "2️⃣ Verificando .gitignore..."
echo ""

# Verificar se .gitignore está ignorando arquivos grandes
if grep -q "outputs/" .gitignore && grep -q "*.pth" .gitignore; then
    echo "  ✅ .gitignore configurado corretamente"
else
    echo "  ⚠️  .gitignore pode não estar ignorando arquivos grandes"
fi

echo ""
echo "3️⃣ Verificando tamanho do repositório..."
echo ""

# Calcular tamanho (excluindo arquivos ignorados)
REPO_SIZE=$(du -sh --exclude=venv --exclude=data --exclude=outputs --exclude=__pycache__ --exclude=.git --exclude=_old_files . | cut -f1)
echo "  📦 Tamanho do repositório: $REPO_SIZE"

if [ -d "outputs" ]; then
    OUTPUTS_SIZE=$(du -sh outputs 2>/dev/null | cut -f1)
    echo "  ⚠️  outputs/: $OUTPUTS_SIZE (será ignorado no git)"
fi

echo ""
echo "4️⃣ Testando instalação limpa..."
echo ""

# Testar se requirements.txt está completo
if python3 -c "import torch, torchvision, matplotlib" 2>/dev/null; then
    echo "  ✅ Dependências principais instaladas"
else
    echo "  ⚠️  Algumas dependências podem estar faltando"
    echo "     Execute: pip install -r requirements.txt"
fi

echo ""
echo "5️⃣ Resumo do que será commitado..."
echo ""

# Mostrar o que será incluído no git
git status --short 2>/dev/null || echo "  ℹ️  Repositório git não inicializado ainda"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Projeto pronto para GitHub!"
echo ""
echo "📋 Próximos passos:"
echo ""
echo "1. Revisar arquivos que serão commitados:"
echo "   git status"
echo ""
echo "2. Adicionar arquivos principais:"
echo "   git add train.py generate.py quick_generate.py"
echo "   git add models.py config.py utils.py"
echo "   git add requirements.txt README.md TRAINING_GUIDE.md"
echo "   git add .gitignore"
echo ""
echo "3. Fazer commit:"
echo "   git commit -m \"Sistema unificado de treinamento de GANs\""
echo ""
echo "4. Push para GitHub:"
echo "   git push origin main"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  IMPORTANTE sobre modelos pré-treinados:"
echo ""
echo "• Modelos .pth são muito grandes (100MB+) para GitHub"
echo "• Estão sendo ignorados no .gitignore"
echo "• Use GitHub Releases para compartilhar modelos"
echo "• Leia PRETRAINED_MODELS.md para mais informações"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
