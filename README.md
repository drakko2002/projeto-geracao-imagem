# 🎨 Sistema de Treinamento de GANs

Sistema completo e unificado para treinar **Generative Adversarial Networks (GANs)** com múltiplos datasets e arquiteturas. Desenvolvido para ser **fácil de usar**, **portátil** e **pronto para compartilhar**.

## ⚡ Início Ultra-Rápido

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Usar menu interativo (recomendado!)
./run.sh

# OU treinar direto via linha de comando
python train.py --dataset mnist --model dcgan --epochs 25
```

## 📋 Índice

- [Características](#-características)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
  - [Menu Interativo](#1-menu-interativo-recomendado)
  - [Linha de Comando](#2-linha-de-comando)
  - [Gerar Imagens](#3-gerar-imagens)
- [Datasets Disponíveis](#-datasets-disponíveis)
- [Modelos GAN](#-modelos-gan)
- [Exemplos Práticos](#-exemplos-práticos)
- [Parâmetros Avançados](#️-parâmetros-avançados)
- [Estrutura de Saída](#-estrutura-de-saída)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)

## ✨ Características

- 🚀 **Menu interativo** - Configure tudo sem digitar comandos
- 📦 **5 datasets** - CIFAR-10, MNIST, Fashion-MNIST, CelebA, Custom
- 🤖 **2 arquiteturas GAN** - DCGAN e WGAN-GP
- 💾 **Checkpoints automáticos** - Retome treinamento a qualquer momento
- 📊 **Visualização em tempo real** - Perdas e amostras geradas
- ⚡ **Suporte GPU/CPU** - Detecta CUDA automaticamente
- 🎯 **Download automático** - Datasets baixados automaticamente
- 📝 **Logs detalhados** - Acompanhe todo o processo

## 📥 Instalação

```bash
# 1. Clonar repositório
git clone https://github.com/seu-usuario/projeto-geracao-imagem.git
cd projeto-geracao-imagem

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Dar permissão ao script (Linux/Mac)
chmod +x run.sh

# 4. Verificar instalação
python train.py --list-datasets
```

### Requisitos

- Python 3.8+
- PyTorch 2.0+
- CUDA (opcional, mas recomendado para GPU)
- 4GB+ RAM (CPU) ou 4GB+ VRAM (GPU)

## 🚀 Como Usar

### 1. Menu Interativo (Recomendado!)

O jeito mais fácil de usar o sistema:

```bash
./run.sh
```

O menu permite:

- ✅ Treinar novos modelos (com assistente passo-a-passo)
- ✅ Gerar imagens de modelos existentes
- ✅ Listar datasets e modelos disponíveis
- ✅ Ver status do treinamento
- ✅ Configurar tudo de forma intuitiva

**Exemplo de uso:**

1. Execute `./run.sh`
2. Escolha opção `1` (Treinar novo modelo)
3. Selecione dataset (ex: `2` para MNIST)
4. Selecione modelo (ex: `1` para DCGAN)
5. Configure épocas (ex: `25`)
6. Confirme e deixe treinar!

### 2. Linha de Comando

Para usuários avançados ou automação:

```bash
# Sintaxe básica
python train.py --dataset <dataset> --model <modelo> --epochs <num>

# Exemplo: MNIST com DCGAN
python train.py --dataset mnist --model dcgan --epochs 25

# Exemplo: CIFAR-10 com WGAN-GP
python train.py --dataset cifar10 --model wgan-gp --epochs 100

# Ver todas as opções
python train.py --help
```

### 3. Gerar Imagens

Após treinar, gere imagens do seu modelo:

#### Opção A: Modo automático (mais fácil)

```bash
python quick_generate.py
```

- Encontra automaticamente o último modelo treinado
- Pergunta quantas imagens gerar
- Salva no mesmo diretório do modelo

#### Opção B: Especificar checkpoint

```bash
python generate.py \
  --checkpoint outputs/mnist/dcgan_20241107_120540/checkpoints/checkpoint_latest.pth \
  --num-samples 64 \
  --output minha_imagem.png
```

#### Opção C: Via menu interativo

```bash
./run.sh
# Escolha opção 2 (Gerar imagens)
```

## 📦 Datasets Disponíveis

| Dataset           | Descrição                                     | Imagens  | Download     | Tamanho  | Canais |
| ----------------- | --------------------------------------------- | -------- | ------------ | -------- | ------ |
| **CIFAR-10**      | 10 categorias coloridas (aviões, carros, etc) | 60.000   | ✅ Automático | 32x32    | RGB    |
| **MNIST**         | Dígitos 0-9 escritos à mão                    | 70.000   | ✅ Automático | 28x28    | Gray   |
| **Fashion-MNIST** | Roupas e acessórios (10 categorias)           | 70.000   | ✅ Automático | 28x28    | Gray   |
| **CelebA**        | Faces de celebridades                         | ~200.000 | ⚠️ Manual     | 178x218  | RGB    |
| **Custom**        | Suas próprias imagens                         | Variável | 📁 Local      | Variável | RGB    |

### Ver lista completa

```bash
python train.py --list-datasets
```

### Usar dataset customizado

### 1. Organize suas imagens

```bash
data/
└── custom/
    └── sua_categoria/
        ├── imagem1.jpg
        ├── imagem2.png
        └── ...
```

### 2. Treine

```bash
python train.py --dataset custom --model dcgan --epochs 100
```

## 🤖 Modelos GAN

### 1. DCGAN (Deep Convolutional GAN)

> **Recomendado para: Iniciantes, treinamento rápido**

```bash
python train.py --dataset mnist --model dcgan --epochs 25
```

**Características:**

- ✅ Estável e fácil de treinar
- ✅ Bons resultados com configurações padrão
- ✅ Mais rápido (~2x que WGAN-GP)
- 📄 Paper: [Radford et al., 2015](https://arxiv.org/abs/1511.06434)

**Configurações padrão:**

- Learning rate: `0.0002`
- Beta1: `0.5`
- Otimizador: Adam

### 2. WGAN-GP (Wasserstein GAN + Gradient Penalty)

Recomendado para: Melhor qualidade, projetos sérios

```bash
python train.py --dataset cifar10 --model wgan-gp --epochs 100
```

**Características:**

- ✅ Treinamento mais estável
- ✅ Menos mode collapse
- ✅ Melhor qualidade de imagens
- ⚠️ Mais lento (5x treino do discriminador)
- 📄 Paper: [Gulrajani et al., 2017](https://arxiv.org/abs/1704.00028)

**Configurações padrão:**

- Learning rate: `0.0001`
- Beta1: `0.0`
- N_critic: `5` (treina critic 5x por batch)
- Lambda_gp: `10.0` (gradient penalty)

### Ver lista completa de modelos

```bash
python train.py --list-models
```

## 💡 Exemplos Práticos

### 🎯 Teste Rápido (5 minutos)

```bash
python train.py --dataset mnist --model dcgan --epochs 5 --batch-size 128
```

### 🚀 Treinamento Básico (30 minutos)

```bash
python train.py --dataset mnist --model dcgan --epochs 25
```

### 🎨 Qualidade Média (1-2 horas)

```bash
python train.py --dataset cifar10 --model dcgan --epochs 50 --batch-size 64
```

### ⭐ Alta Qualidade (3-5 horas)

```bash
python train.py --dataset cifar10 --model wgan-gp --epochs 200 --batch-size 64
```

### 🖼️ Imagens de Alta Resolução 128px (2-3 horas)

```bash
python train.py --dataset cifar10 --model dcgan --img-size 128 --epochs 50 --batch-size 64
```

### 🖼️ Imagens de Altíssima Resolução 256px (5+ horas)

```bash
python train.py --dataset celeba --model dcgan --img-size 256 --ngf 128 --ndf 128 --epochs 100 --batch-size 32
```

### 💾 GPU com Pouca Memória (RTX 4060 8GB)

```bash
# 128px - batch-size 64 recomendado
python train.py --dataset fashion-mnist --model dcgan --img-size 128 --batch-size 64 --workers 2

# 256px - batch-size 32 recomendado
python train.py --dataset cifar10 --model dcgan --img-size 256 --batch-size 32 --ngf 96 --ndf 96
```

### 📁 Dataset Customizado

```bash
python train.py --dataset custom --model dcgan --epochs 100 --img-size 64
```

## ⚙️ Parâmetros Avançados

### Parâmetros Principais

```bash
python train.py \
  --dataset <nome>         # Dataset: cifar10, mnist, fashion-mnist, celeba, custom
  --model <nome>           # Modelo: dcgan, wgan-gp
  --epochs <num>           # Número de épocas (padrão: 50)
  --batch-size <num>       # Tamanho do batch (padrão: 128)
  --img-size <num>         # Tamanho da imagem (padrão: 128, presets: 128/256, mínimo: 128)
  --lr <float>             # Learning rate (auto-detectado se omitido)
  --nz <num>               # Dimensão do vetor latente (padrão: 100)
  --ngf <num>              # Filtros do gerador (padrão: 64, use 96-128 para 256px)
  --ndf <num>              # Filtros do discriminador (padrão: 64, use 96-128 para 256px)
  --workers <num>          # Workers do DataLoader (padrão: 2)
  --ngpu <num>             # Número de GPUs (padrão: 1)
```

### Exemplos de Configurações

#### Aumentar capacidade do modelo

```bash
--ngf 128 --ndf 128  # Mais filtros = mais capacidade
```

#### Ajustar learning rate

```bash
--lr 0.0001  # Menor = mais estável, mais lento
--lr 0.0005  # Maior = mais rápido, menos estável
```

#### Usar múltiplas GPUs

```bash
--ngpu 2  # Usar 2 GPUs
```

#### Processar mais dados em paralelo

```bash
--workers 4  # Mais workers = carregamento mais rápido
```

### Ver todas as opções

```bash
python train.py --help
```

## 📂 Estrutura de Saída

Após o treinamento, os resultados são salvos em `outputs/`:

```bash
outputs/
└── <dataset>/
    └── <modelo>_<timestamp>/
        ├── config.json              # ⚙️ Configurações usadas
        ├── training.log             # 📝 Log completo do treinamento
        ├── training_losses.png      # 📊 Gráfico de perdas
        ├── final_samples.png        # 🎨 Imagens finais geradas
        ├── samples/                 # 📸 Amostras por época
        │   ├── epoch_5.png
        │   ├── epoch_10.png
        │   └── ...
        └── checkpoints/             # 💾 Modelos salvos
            ├── checkpoint_epoch_10.pth
            ├── checkpoint_epoch_20.pth
            └── checkpoint_latest.pth  # ⭐ Último checkpoint
```

### Exemplo real

```bash
outputs/mnist/dcgan_20241107_120540/
├── config.json                    # Hiperparâmetros usados
├── training.log                   # "Epoch 1/25, Loss_D: 0.5, Loss_G: 1.2, ..."
├── training_losses.png            # Gráfico D_loss vs G_loss
├── final_samples.png              # Grid 8x8 de imagens geradas
├── samples/
│   ├── epoch_5.png               # Como estava na época 5
│   ├── epoch_10.png
│   └── epoch_25.png
└── checkpoints/
    ├── checkpoint_epoch_10.pth   # Checkpoint da época 10 (75MB)
    ├── checkpoint_epoch_20.pth   # Checkpoint da época 20 (75MB)
    └── checkpoint_latest.pth     # Checkpoint final (75MB)
```

### O que cada checkpoint contém

- ✅ Pesos completos do gerador
- ✅ Pesos completos do discriminador
- ✅ Estados dos otimizadores
- ✅ Configurações do modelo
- ✅ Histórico de perdas
- ✅ Época atual

**Você pode retomar o treinamento de qualquer checkpoint!**

## 💾 Checkpoints - Guia Completo

### 📍 Onde ficam os checkpoints?

Os checkpoints são salvos automaticamente durante o treinamento em:

```
outputs/<dataset>/<modelo>_<timestamp>/checkpoints/
```

Exemplo:
```
outputs/mnist/dcgan_20241207_143022/checkpoints/
├── checkpoint_epoch_10.pth      # Salvo na época 10
├── checkpoint_epoch_20.pth      # Salvo na época 20
└── checkpoint_latest.pth        # Sempre o mais recente ⭐
```

### 🚀 Como usar checkpoints para geração

**Método 1: Automático (recomendado)**

```bash
python quick_generate.py
# Encontra automaticamente o último checkpoint e gera imagens
```

**Método 2: Especificar checkpoint**

```bash
python generate.py \
  --checkpoint outputs/mnist/dcgan_20241207_143022/checkpoints/checkpoint_latest.pth \
  --num-samples 64 \
  --upscale 2x
```

**Método 3: Geração interativa com upscale**

```bash
python generate_interactive.py \
  --checkpoint outputs/cifar10/dcgan_xxx/checkpoints/checkpoint_latest.pth \
  --upscale 8 \
  --upscale-method lanczos
```

### 🎨 Opções de upscale na geração

Todos os scripts de geração agora suportam upscaling pós-processamento:

```bash
# Sem upscale (padrão no generate.py)
python generate.py --checkpoint <path> --upscale none

# Upscale 2x
python generate.py --checkpoint <path> --upscale 2x

# Upscale 4x com método bicubic
python generate.py --checkpoint <path> --upscale 4x --upscale-method bicubic

# Upscale 8x com lanczos (melhor qualidade)
python generate.py --checkpoint <path> --upscale 8x --upscale-method lanczos
```

**Métodos disponíveis:**
- `lanczos` - Melhor qualidade (padrão)
- `bicubic` - Rápido e bom
- `nearest` - Pixel-perfect (estilo retro)

### 🔄 Como retomar treinamento (futura implementação)

```bash
# Retomar do último checkpoint
python train.py --resume outputs/mnist/dcgan_xxx/checkpoints/checkpoint_latest.pth

# Retomar de época específica
python train.py --resume outputs/mnist/dcgan_xxx/checkpoints/checkpoint_epoch_20.pth
```

> **Nota:** A funcionalidade de retomar treinamento será implementada em breve.

### 📦 Como transportar para outra máquina

**Passo 1: Preparar para transporte**

```bash
# Criar pacote com checkpoint e config
cd outputs/mnist/dcgan_20241207_143022
zip -r meu_modelo.zip checkpoints/checkpoint_latest.pth config.json

# Ou copiar apenas o essencial
cp checkpoints/checkpoint_latest.pth ~/modelo_mnist.pth
cp config.json ~/modelo_mnist_config.json
```

**Passo 2: Na máquina de destino**

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Criar estrutura de diretórios
mkdir -p outputs/mnist/modelo_importado/checkpoints

# 3. Copiar arquivos
cp modelo_mnist.pth outputs/mnist/modelo_importado/checkpoints/checkpoint_latest.pth
cp modelo_mnist_config.json outputs/mnist/modelo_importado/config.json

# 4. Gerar imagens
python generate.py \
  --checkpoint outputs/mnist/modelo_importado/checkpoints/checkpoint_latest.pth \
  --num-samples 64
```

### 💡 Dicas de portabilidade

✅ **O que levar:**
- `checkpoint_latest.pth` (essencial) - ~50-150MB
- `config.json` (essencial) - <1KB
- `training_losses.png` (opcional) - Histórico visual
- `final_samples.png` (opcional) - Exemplos de saída

✅ **Sistemas compatíveis:**
- Windows, Linux, macOS
- GPU (CUDA) ou CPU
- Python 3.8+

✅ **Compartilhamento:**
- GitHub Releases (<100MB)
- Google Drive / Dropbox
- Hugging Face Hub (recomendado para >100MB)

### 📊 Tamanho dos checkpoints

| Modelo | Resolução | ngf/ndf | Tamanho aprox. |
|--------|-----------|---------|----------------|
| DCGAN  | 128px     | 64      | ~50MB          |
| DCGAN  | 256px     | 64      | ~50MB          |
| DCGAN  | 256px     | 128     | ~150MB         |
| WGAN-GP| 128px     | 64      | ~50MB          |
| WGAN-GP| 256px     | 128     | ~150MB         |

### 🎯 Exemplo completo: Compartilhar modelo treinado

```bash
# 1. Na máquina de origem (após treinar)
cd outputs/mnist/dcgan_20241207_143022
zip -r mnist_dcgan_trained.zip \
  checkpoints/checkpoint_latest.pth \
  config.json \
  final_samples.png \
  training_losses.png

# 2. Compartilhar mnist_dcgan_trained.zip (GitHub, Drive, etc)

# 3. Na máquina de destino
unzip mnist_dcgan_trained.zip -d imported_model/

# 4. Gerar imagens
python generate.py \
  --checkpoint imported_model/checkpoints/checkpoint_latest.pth \
  --num-samples 100 \
  --upscale 4x
```

## 🔧 Troubleshooting

### ❌ "CUDA out of memory"

**Solução:** Reduza batch size ou tamanho da imagem

```bash
python train.py --dataset mnist --model dcgan --batch-size 32 --img-size 32
```

### ❌ "No module named 'torch'"

**Solução:** Instale PyTorch

```bash
pip install torch torchvision
```

### ❌ "RuntimeError: CUDA not available"

**Solução:** Treine na CPU (mais lento, mas funciona)

```bash
# O código detecta automaticamente e usa CPU
python train.py --dataset mnist --model dcgan --epochs 10
```

### ❌ "FileNotFoundError: data/custom not found"

**Solução:** Crie a estrutura de pastas correta

```bash
mkdir -p data/custom/sua_categoria
# Coloque suas imagens em data/custom/sua_categoria/
```

### ❌ Treinamento muito lento

**Soluções:**

```bash
# 1. Use GPU se disponível
nvidia-smi  # Verifica se GPU está disponível

# 2. Reduza epochs para testes
python train.py --dataset mnist --model dcgan --epochs 5

# 3. Use dataset menor
python train.py --dataset mnist --model dcgan  # Mais rápido que cifar10

# 4. Use DCGAN em vez de WGAN-GP
python train.py --dataset cifar10 --model dcgan  # 2x mais rápido
```

### ❌ Imagens geradas ruins

**Soluções:**

```bash
# 1. Treine por mais épocas
python train.py --dataset mnist --model dcgan --epochs 50

# 2. Use WGAN-GP para melhor qualidade
python train.py --dataset mnist --model wgan-gp --epochs 100

# 3. Ajuste learning rate
python train.py --dataset mnist --model dcgan --lr 0.0001

# 4. Aumente capacidade do modelo
python train.py --dataset mnist --model dcgan --ngf 128 --ndf 128
```

### ❌ "Mode collapse" (imagens todas iguais)

**Solução:** Use WGAN-GP

```bash
python train.py --dataset cifar10 --model wgan-gp --epochs 100
```

## ❓ FAQ

### Q: Quanto tempo leva para treinar?

**A:** Depende do dataset e hardware:

- **MNIST (DCGAN, GPU):** ~10-15 minutos (25 épocas)
- **CIFAR-10 (DCGAN, GPU):** ~1-2 horas (50 épocas)
- **CIFAR-10 (WGAN-GP, GPU):** ~3-5 horas (100 épocas)
- **CelebA (DCGAN, GPU):** ~5-8 horas (100 épocas)
- **CPU:** ~10-20x mais lento que GPU

### Q: Preciso de GPU?

**A:** Não é obrigatório, mas **fortemente recomendado**:

- ✅ GPU: Treinamento em horas
- ❌ CPU: Treinamento em dias

### Q: Qual modelo usar?

**A:**

- **Iniciante/Teste:** DCGAN (mais rápido, mais fácil)
- **Qualidade/Produção:** WGAN-GP (melhor resultado, mais lento)

### Q: Quantas épocas treinar?

**A:** Recomendações:

- **MNIST:** 25-50 épocas
- **Fashion-MNIST:** 50-75 épocas
- **CIFAR-10:** 50-100 épocas (DCGAN) ou 100-200 (WGAN-GP)
- **CelebA:** 100-200 épocas

### Q: Como usar minhas próprias imagens?

**A:**

1. Crie pasta: `data/custom/categoria/`
2. Coloque suas imagens (.jpg, .png)
3. Execute: `python train.py --dataset custom --model dcgan --epochs 100`
4. Recomendado: 10.000+ imagens para bons resultados

### Q: Posso retomar um treinamento interrompido?

**A:** Sim! (em desenvolvimento - será adicionado em breve)

### Q: Como compartilhar meu modelo treinado?

**A:**

1. **Compactar checkpoint:**

   ```bash
   cd outputs/mnist/dcgan_xxx/checkpoints/
   zip meu_modelo.zip checkpoint_latest.pth
   ```

2. **Compartilhar via:**
   - GitHub Releases (recomendado para <2GB)
   - Google Drive / Dropbox
   - Hugging Face Hub

3. **Outros podem usar:**

   ```bash
   python generate.py --checkpoint checkpoint_latest.pth --num-samples 100
   ```

### Q: Qual tamanho de batch usar?

**A:** Depende da memória da GPU e resolução:

**Para 128px (padrão):**
- **16GB+ VRAM:** batch-size 128-256
- **8GB VRAM (RTX 4060):** batch-size 64-128
- **4GB VRAM:** batch-size 32-64
- **CPU:** batch-size 32

**Para 256px:**
- **16GB+ VRAM:** batch-size 64-128
- **8GB VRAM (RTX 4060):** batch-size 32-64
- **4GB VRAM:** batch-size 16-32
- **CPU:** batch-size 16

### Q: O que é "mode collapse"?

**A:** Quando o gerador produz sempre as mesmas imagens. **Solução:** Use WGAN-GP.

### Q: Como melhorar a qualidade das imagens?

**A:**

1. Treine por mais épocas
2. Use WGAN-GP em vez de DCGAN
3. Aumente capacidade: `--ngf 128 --ndf 128`
4. Use dataset maior e de melhor qualidade
5. Ajuste learning rate: `--lr 0.0001`

## 📚 Recursos de Aprendizado

- 📄 **DCGAN Paper:** <https://arxiv.org/abs/1511.06434>
- 📄 **WGAN-GP Paper:** <https://arxiv.org/abs/1704.00028>
- 📖 **PyTorch Tutorials:** <https://pytorch.org/tutorials/>
- 🎓 **GANs Course:** <https://www.coursera.org/learn/generative-adversarial-networks-gans>

## 🤝 Contribuindo

Contribuições são bem-vindas!

Para adicionar:

- **Novo dataset:** Edite `config.py` → função `get_dataset()`
- **Novo modelo:** Edite `models.py` → adicione classe do modelo
- **Nova feature:** Abra um Pull Request

## 📝 Licença

Projeto open source - Use e modifique livremente!

## 🎯 Próximos Passos

1. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Execute o menu interativo:**

   ```bash
   ./run.sh
   ```

3. **Ou faça seu primeiro treinamento:**

   ```bash
   python train.py --dataset mnist --model dcgan --epochs 25
   ```

4. **Gere imagens:**

   ```bash
   python quick_generate.py
   ```

5. **Experimente outros datasets e modelos!**

## 📞 Suporte

- 🐛 **Bug?** Abra uma [issue](https://github.com/seu-usuario/projeto-geracao-imagem/issues)
- 💡 **Sugestão?** Abra uma [discussion](https://github.com/seu-usuario/projeto-geracao-imagem/discussions)
- ⭐ **Gostou?** Dê uma estrela no projeto!

> **Bom treinamento! 🚀🎨**
