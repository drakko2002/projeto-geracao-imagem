#!/bin/bash
# Script de limpeza e organização do projeto

echo "🧹 Limpando e organizando o projeto..."
echo ""

# Criar pasta para arquivos antigos (backup)
mkdir -p _old_files

# Mover arquivos antigos/desnecessários
echo "📦 Movendo arquivos antigos para _old_files/..."

# Stable Diffusion (não relacionado ao projeto GAN)
[ -f app.py ] && mv app.py _old_files/
[ -f download_model.py ] && mv download_model.py _old_files/
[ -f run.sh ] && mv run.sh _old_files/
[ -f Dockerfile ] && mv Dockerfile _old_files/

# Pastas antigas com código redundante
[ -d dcgan ] && mv dcgan _old_files/
[ -d scripts ] && mv scripts _old_files/
[ -d src ] && mv src _old_files/
[ -d test ] && mv test _old_files/

# Arquivos de documentação redundantes
[ -f EXAMPLES.txt ] && mv EXAMPLES.txt _old_files/
[ -f SUMMARY.md ] && mv SUMMARY.md _old_files/

# Testes (úteis para dev, mas não para usuários finais)
[ -f test_models.py ] && mv test_models.py _old_files/
[ -f test_system.py ] && mv test_system.py _old_files/

echo ""
echo "✅ Arquivos movidos para _old_files/"
echo ""

# Atualizar README se necessário
if [ -f README_NOVO.md ]; then
    echo "📝 Substituindo README.md..."
    mv README.md _old_files/README_OLD.md
    mv README_NOVO.md README.md
fi

echo ""
echo "📂 Estrutura atual do projeto:"
echo ""
tree -L 2 -I 'venv|__pycache__|data|outputs|_old_files|.git' .

echo ""
echo "✨ Limpeza concluída!"
echo ""
echo "Arquivos principais mantidos:"
echo "  ✅ train.py - Treinamento"
echo "  ✅ generate.py - Geração de imagens"  
echo "  ✅ quick_generate.py - Helper de geração"
echo "  ✅ models.py - Arquiteturas GAN"
echo "  ✅ config.py - Configurações"
echo "  ✅ utils.py - Utilitários"
echo "  ✅ requirements.txt - Dependências"
echo "  ✅ README.md - Documentação"
echo "  ✅ TRAINING_GUIDE.md - Guia completo"
echo "  ✅ quickstart.sh - Menu interativo"
echo ""
echo "Arquivos antigos em: _old_files/"
echo ""
