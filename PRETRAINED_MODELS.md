# 📦 Modelos Pré-Treinados

Este arquivo explica como compartilhar e usar modelos pré-treinados.

## ⚠️ Problema: Modelos são Grandes Demais para GitHub

Checkpoints de modelos GAN geralmente têm **100MB-500MB+**, o que excede o limite do GitHub (100MB por arquivo).

## ✅ Solução: Três Opções

### Opção 1: Git LFS (Recomendado para poucos modelos)

Use Git Large File Storage para arquivos grandes:

```bash
# Instalar Git LFS
git lfs install

# Rastrear arquivos .pth
git lfs track "*.pth"
git lfs track "outputs/**/*.pth"

# Commit do .gitattributes
git add .gitattributes
git commit -m "Configurar Git LFS para modelos"

# Agora pode commitar modelos normalmente
git add outputs/mnist/dcgan_xxx/checkpoints/checkpoint_latest.pth
git commit -m "Adicionar modelo MNIST pré-treinado"
git push
```

**Limitação:** GitHub LFS tem limite de 1GB grátis/mês de bandwidth.

---

### Opção 2: GitHub Releases (Recomendado para vários modelos)

Crie releases com os modelos anexados:

```bash
# 1. Criar release no GitHub
# 2. Anexar arquivo .pth ou .zip com modelos
# 3. Usuários baixam da página de releases
```

**Exemplo de uso:**
```bash
# Baixar modelo da release
wget https://github.com/seu-usuario/projeto-geracao-imagem/releases/download/v1.0/mnist_dcgan.pth -O checkpoint.pth

# Gerar imagens
python generate.py --checkpoint checkpoint.pth
```

---

### Opção 3: Serviço Externo (Melhor para muitos modelos)

Use Google Drive, Dropbox, Hugging Face Model Hub, etc:

```bash
# Google Drive
# 1. Upload do modelo
# 2. Tornar link público
# 3. Compartilhar link

# Usuários baixam com:
# (Adicione script helper para download automático)
```

---

## 🎯 Recomendação para Este Projeto

**Para começar:**

1. **Mantenha modelos FORA do git** (já configurado no .gitignore)
2. **Inclua no README:** Link para baixar modelos pré-treinados
3. **Use GitHub Releases** para disponibilizar modelos

**Estrutura recomendada para releases:**

```
v1.0-models/
├── mnist_dcgan_epoch50.pth          (~100MB)
├── cifar10_dcgan_epoch100.pth       (~150MB)
├── fashion_mnist_wgan_epoch100.pth  (~120MB)
└── README.txt                       (instruções)
```

---

## 📝 Instruções para Usuários

Adicione isto ao README.md principal:

```markdown
## 🚀 Usando Modelos Pré-Treinados

### Download
Baixe modelos pré-treinados da [página de releases](link):

- MNIST + DCGAN (50 épocas) - 100MB
- CIFAR-10 + DCGAN (100 épocas) - 150MB
- Fashion-MNIST + WGAN-GP (100 épocas) - 120MB

### Uso
\`\`\`bash
# Baixar modelo
wget <link-do-modelo> -O modelo.pth

# Gerar imagens
python generate.py --checkpoint modelo.pth --num-samples 100
\`\`\`
```

---

## 🛠️ Script Helper (Futuro)

Criar `download_models.py` para baixar automaticamente:

```python
#!/usr/bin/env python3
"""Download modelos pré-treinados"""

MODELS = {
    'mnist-dcgan': 'https://github.com/.../mnist_dcgan.pth',
    'cifar10-dcgan': 'https://github.com/.../cifar10_dcgan.pth',
}

# Implementar download automático
```

---

## 📊 Tamanho Estimado dos Modelos

| Modelo  | Dataset       | Tamanho Aproximado |
| ------- | ------------- | ------------------ |
| DCGAN   | MNIST         | ~100 MB            |
| DCGAN   | CIFAR-10      | ~150 MB            |
| DCGAN   | Fashion-MNIST | ~100 MB            |
| WGAN-GP | CIFAR-10      | ~150 MB            |

**Total:** ~500MB-1GB para todos os modelos

---

## ✅ Decisão Final

Para este projeto, **recomendo**:

1. ✅ Manter `outputs/` no .gitignore
2. ✅ Criar GitHub Release com 1-2 modelos exemplo
3. ✅ Adicionar instruções no README de como baixar
4. ✅ Focar em facilitar treinamento (usuários treinam próprios modelos)

**Foco:** Código limpo e fácil de usar > Distribuir modelos pesados
