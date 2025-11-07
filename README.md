# 🎨 Projeto de Geração de Imagens com GANs

Sistema completo e unificado para treinar **Generative Adversarial Networks (GANs)** com múltiplos datasets e arquiteturas diferentes. Desenvolvido para ser **fácil de usar**, **portátil** e **pronto para compartilhar**.

## ✨ Características

- 🚀 **Treinamento simplificado** em um único arquivo
- 📦 **5 datasets suportados** (download automático incluído)
- 🤖 **2 arquiteturas GAN** (DCGAN e WGAN-GP)
- 💾 **Sistema de checkpoints** para retomar treinamento
- 📊 **Visualização automática** de perdas e amostras
- 🎯 **Configuração flexível** via CLI ou arquivos
- 📝 **Logs detalhados** de todo o processo
- ⚡ **Suporte para GPU** (CUDA) e CPU

## 🎯 Início Rápido

### 1. Instalação

```bash
# Clonar repositório
git clone <seu-repositorio>
cd projeto-geracao-imagem

# Instalar dependências
pip install -r requirements.txt
```

### 2. Treinar seu primeiro modelo

```bash
# Treinar DCGAN com CIFAR-10 (download automático)
python train.py --dataset cifar10 --model dcgan --epochs 50

# Treinar WGAN-GP com Fashion-MNIST
python train.py --dataset fashion-mnist --model wgan-gp --epochs 100

# Treinar com MNIST (rápido para testes)
python train.py --dataset mnist --model dcgan --epochs 25
```

### 3. Gerar imagens

```bash
# Gerar imagens usando modelo treinado
python generate.py --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth --num-samples 64
```

## 📦 Datasets Suportados

| Dataset           | Descrição                | Download       | Imagens  |
| ----------------- | ------------------------ | -------------- | -------- |
| **CIFAR-10**      | 10 categorias coloridas  | ✅ Automático   | 60.000   |
| **MNIST**         | Dígitos 0-9 em grayscale | ✅ Automático   | 70.000   |
| **Fashion-MNIST** | Roupas e acessórios      | ✅ Automático   | 70.000   |
| **CelebA**        | Faces de celebridades    | ⚠️ Manual       | ~200.000 |
| **Custom**        | Suas próprias imagens    | 📁 Suas imagens | Variável |

### Ver todos os datasets

```bash
python train.py --list-datasets
```

## 🤖 Modelos Suportados

| Modelo      | Descrição                          | Velocidade | Qualidade       |
| ----------- | ---------------------------------- | ---------- | --------------- |
| **DCGAN**   | Deep Convolutional GAN             | ⚡ Rápido   | ⭐⭐⭐ Boa         |
| **WGAN-GP** | Wasserstein GAN + Gradient Penalty | 🐢 Lento    | ⭐⭐⭐⭐⭐ Excelente |

### Ver todos os modelos

```bash
python train.py --list-models
```

## 💡 Exemplos de Uso

### Exemplo 1: Treinamento Básico

```bash
python train.py --dataset cifar10 --model dcgan --epochs 50
```

### Exemplo 2: Alta Qualidade (requer mais tempo)

```bash
python train.py --dataset cifar10 --model wgan-gp --epochs 200 --batch-size 64
```

### Exemplo 3: Teste Rápido

```bash
python train.py --dataset mnist --model dcgan --epochs 5
```

### Exemplo 4: Imagens de Alta Resolução

```bash
python train.py --dataset celeba --model dcgan --img-size 128 --ngf 128 --ndf 128 --epochs 100
```

### Exemplo 5: GPU com Pouca Memória

```bash
python train.py --dataset fashion-mnist --model dcgan --batch-size 32
```

### Exemplo 6: Dataset Customizado

```bash
# Organize suas imagens em: data/custom/categoria/
python train.py --dataset custom --model dcgan --epochs 100
```

## ⚙️ Parâmetros Principais

```bash
python train.py \
  --dataset <dataset>      # Dataset: cifar10, mnist, fashion-mnist, celeba, custom
  --model <modelo>         # Modelo: dcgan, wgan-gp
  --epochs <num>           # Número de épocas (padrão: 50)
  --batch-size <num>       # Tamanho do batch (padrão: 128)
  --img-size <num>         # Tamanho das imagens (padrão: 64)
  --lr <float>             # Learning rate (auto se não especificado)
  --nz <num>               # Tamanho vetor latente (padrão: 100)
  --ngf <num>              # Filtros do gerador (padrão: 64)
  --ndf <num>              # Filtros do discriminador (padrão: 64)
  --workers <num>          # Workers DataLoader (padrão: 2)
  --ngpu <num>             # Número de GPUs (padrão: 1)
```

## 📂 Estrutura do Projeto

```bash
projeto-geracao-imagem/
├── train.py              # ⭐ Script principal de treinamento
├── generate.py           # 🎨 Gerar imagens de modelos treinados
├── models.py             # 🤖 Arquiteturas GAN (DCGAN, WGAN-GP)
├── config.py             # ⚙️ Configurações de datasets e modelos
├── utils.py              # 🛠️ Funções auxiliares
├── requirements.txt      # 📦 Dependências
├── TRAINING_GUIDE.md     # 📖 Guia completo de treinamento
└── outputs/              # 📁 Resultados dos treinamentos
    └── <dataset>/
        └── <modelo>_<timestamp>/
            ├── config.json
            ├── training.log
            ├── training_losses.png
            ├── final_samples.png
            ├── samples/
            └── checkpoints/
```

## 📊 Resultados do Treinamento

Após o treinamento, você terá:

```bash
outputs/cifar10/dcgan_20240115_143000/
├── config.json                    # Configurações usadas
├── training.log                   # Log detalhado
├── training_losses.png            # Gráfico de perdas
├── final_samples.png              # Amostras finais
├── samples/                       # Amostras por época
│   ├── epoch_5.png
│   ├── epoch_10.png
│   └── ...
└── checkpoints/                   # Modelos salvos
    ├── checkpoint_epoch_10.pth
    ├── checkpoint_epoch_20.pth
    └── checkpoint_latest.pth
```

## 🎯 Compartilhando Modelos

### Para compartilhar seu modelo treinado

#### 1. **Compactar checkpoint:**

```bash
cd outputs/cifar10/dcgan_xxx/checkpoints/
zip meu_modelo.zip checkpoint_latest.pth
```

#### 2. **Outras pessoas podem usar:**

```bash
# Download do modelo compartilhado
unzip meu_modelo.zip

# Gerar imagens
python generate.py --checkpoint checkpoint_latest.pth --num-samples 100
```

### O checkpoint contém

- ✅ Pesos do gerador
- ✅ Pesos do discriminador
- ✅ Estados dos otimizadores
- ✅ Configurações completas
- ✅ Histórico de perdas

## 🔧 Requisitos

- Python 3.8+
- PyTorch 2.0+
- CUDA (opcional, mas recomendado)
- 4GB+ RAM (CPU) ou 4GB+ VRAM (GPU)

## 📖 Documentação Completa

Para guia detalhado com exemplos, dicas e troubleshooting:

👉 **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** 👈

## 🚀 Próximos Passos

Depois de treinar seu modelo:

1. **Gerar imagens:**

   ```bash
   python generate.py --checkpoint outputs/.../checkpoint_latest.pth
   ```

2. **Compartilhar no GitHub:**
   - Adicione checkpoints ao `.gitignore` (são grandes!)
   - Compartilhe apenas o código
   - Use Git LFS para modelos (opcional)

3. **Experimentar:**
   - Tente diferentes datasets
   - Ajuste hiperparâmetros
   - Compare DCGAN vs WGAN-GP

## 🤝 Contribuindo

Contribuições são bem-vindas! Para adicionar:

- Novos datasets: Edite `config.py`
- Novos modelos: Edite `models.py`
- Melhorias: Abra um Pull Request

## 📝 Licença

Open source - Use e modifique livremente!

## 🎓 Recursos de Aprendizado

- **DCGAN Paper:** <https://arxiv.org/abs/1511.06434>
- **WGAN-GP Paper:** <https://arxiv.org/abs/1704.00028>
- **PyTorch Tutorials:** <https://pytorch.org/tutorials/>

## ❓ FAQ

**P: Quanto tempo leva para treinar?**
R: Depende do dataset e GPU. MNIST: ~10min, CIFAR-10: ~1h, CelebA: ~3h (GPU)

**P: Preciso de GPU?**
R: Não é obrigatório, mas recomendado. CPU é muito mais lento.

**P: Qual modelo usar?**
R: DCGAN para começar, WGAN-GP para melhor qualidade.

**P: Quantas épocas são necessárias?**
R: MNIST: 25, CIFAR-10: 50-100, CelebA: 100-200

**P: Como usar meu próprio dataset?**
R: Organize em `data/custom/categoria/` e use `--dataset custom`

---

**Dúvidas?** Abra uma issue! 🚀

**Gostou?** Dê uma ⭐ no projeto!
