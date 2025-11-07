# 📖 Guia de Treinamento de GANs

Este guia explica como treinar modelos GAN (Generative Adversarial Networks) com diferentes datasets e arquiteturas usando este projeto.

## 📋 Índice

- [Instalação](#instalação)
- [Início Rápido](#início-rápido)
- [Datasets Disponíveis](#datasets-disponíveis)
- [Modelos Disponíveis](#modelos-disponíveis)
- [Exemplos de Uso](#exemplos-de-uso)
- [Configurações Avançadas](#configurações-avançadas)
- [Estrutura de Saída](#estrutura-de-saída)
- [Dicas e Boas Práticas](#dicas-e-boas-práticas)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Instalação

### 1. Clonar o repositório
```bash
git clone <seu-repositorio>
cd projeto-geracao-imagem
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Verificar instalação
```bash
python train.py --list-datasets
python train.py --list-models
```

---

## ⚡ Início Rápido

### Treinamento básico com CIFAR-10
```bash
python train.py --dataset cifar10 --model dcgan --epochs 50
```

Isso irá:
- ✅ Baixar automaticamente o dataset CIFAR-10
- ✅ Criar e treinar um modelo DCGAN
- ✅ Salvar checkpoints a cada 10 épocas
- ✅ Gerar amostras de imagens a cada 5 épocas
- ✅ Criar gráficos de perda
- ✅ Salvar logs detalhados

---

## 📦 Datasets Disponíveis

### 1. CIFAR-10
**Imagens coloridas 32x32 de 10 categorias**

```bash
python train.py --dataset cifar10 --model dcgan --epochs 50
```

- **Classes:** Aviões, Carros, Pássaros, Gatos, Cervos, Cachorros, Sapos, Cavalos, Navios, Caminhões
- **Tamanho:** 60.000 imagens (50.000 treino + 10.000 teste)
- **Canais:** RGB (3)
- **Download:** Automático ✅

---

### 2. MNIST
**Dígitos escritos à mão 28x28 em escala de cinza**

```bash
python train.py --dataset mnist --model dcgan --epochs 25
```

- **Classes:** Dígitos de 0-9
- **Tamanho:** 70.000 imagens (60.000 treino + 10.000 teste)
- **Canais:** Grayscale (1)
- **Download:** Automático ✅

---

### 3. Fashion-MNIST
**Imagens 28x28 de roupas e acessórios**

```bash
python train.py --dataset fashion-mnist --model dcgan --epochs 50
```

- **Classes:** Camiseta, Calça, Suéter, Vestido, Casaco, Sandália, Camisa, Tênis, Bolsa, Bota
- **Tamanho:** 70.000 imagens
- **Canais:** Grayscale (1)
- **Download:** Automático ✅

---

### 4. CelebA
**Imagens de celebridades (requer download manual)**

```bash
python train.py --dataset celeba --model dcgan --epochs 100 --img-size 128
```

- **Classes:** Faces de celebridades
- **Tamanho:** ~200.000 imagens
- **Canais:** RGB (3)
- **Download:** Manual ⚠️

**Como obter:**
1. Baixe de: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Extraia para: `./data/celeba/`
3. Organize em subpastas (ex: `./data/celeba/img_align_celeba/`)

---

### 5. Custom Dataset
**Use suas próprias imagens!**

```bash
python train.py --dataset custom --model dcgan --epochs 100
```

**Estrutura de pastas necessária:**
```
data/
└── custom/
    └── sua_categoria/
        ├── imagem1.jpg
        ├── imagem2.png
        └── ...
```

- **Formatos suportados:** JPG, PNG, etc.
- **Recomendação:** Pelo menos 10.000 imagens para bons resultados
- **Canais:** RGB (3)

---

## 🤖 Modelos Disponíveis

### 1. DCGAN (Deep Convolutional GAN)
**Recomendado para iniciantes**

```bash
python train.py --dataset cifar10 --model dcgan --epochs 50
```

**Características:**
- ✅ Estável e fácil de treinar
- ✅ Bons resultados com configurações padrão
- ✅ Mais rápido
- 📄 Paper: Radford et al., 2015

**Configurações padrão:**
- Learning rate: 0.0002
- Beta1: 0.5
- Otimizador: Adam

---

### 2. WGAN-GP (Wasserstein GAN with Gradient Penalty)
**Recomendado para resultados de alta qualidade**

```bash
python train.py --dataset cifar10 --model wgan-gp --epochs 100
```

**Características:**
- ✅ Treinamento mais estável
- ✅ Menos mode collapse
- ✅ Melhor qualidade de imagens
- ⚠️ Mais lento (treina discriminador 5x por época)
- 📄 Paper: Gulrajani et al., 2017

**Configurações padrão:**
- Learning rate: 0.0001
- Beta1: 0.0
- N_critic: 5 (treina critic 5 vezes por iteração do gerador)
- Lambda_gp: 10.0 (peso do gradient penalty)

---

## 💡 Exemplos de Uso

### Exemplo 1: Treinamento rápido para testes
```bash
python train.py --dataset mnist --model dcgan --epochs 5 --batch-size 128
```

### Exemplo 2: Treinamento de alta qualidade
```bash
python train.py --dataset cifar10 --model wgan-gp --epochs 200 --img-size 64 --batch-size 64
```

### Exemplo 3: Imagens de alta resolução
```bash
python train.py --dataset celeba --model dcgan --epochs 100 --img-size 128 --ngf 128 --ndf 128
```

### Exemplo 4: Treinamento com GPU limitada
```bash
python train.py --dataset fashion-mnist --model dcgan --epochs 50 --batch-size 32 --workers 1
```

### Exemplo 5: Learning rate customizado
```bash
python train.py --dataset cifar10 --model dcgan --epochs 50 --lr 0.0001 --beta1 0.5
```

---

## ⚙️ Configurações Avançadas

### Parâmetros principais

| Parâmetro      | Descrição                | Padrão | Recomendação                     |
| -------------- | ------------------------ | ------ | -------------------------------- |
| `--epochs`     | Número de épocas         | 50     | 50-200 dependendo do dataset     |
| `--batch-size` | Tamanho do batch         | 128    | 64-128 (menor se pouca GPU RAM)  |
| `--img-size`   | Tamanho das imagens      | 64     | 64 (básico), 128 (avançado)      |
| `--lr`         | Learning rate            | auto   | 0.0002 (DCGAN), 0.0001 (WGAN-GP) |
| `--nz`         | Tamanho vetor latente    | 100    | 100-512                          |
| `--ngf`        | Filtros do gerador       | 64     | 64-128                           |
| `--ndf`        | Filtros do discriminador | 64     | 64-128                           |
| `--workers`    | Workers DataLoader       | 2      | 2-4                              |
| `--ngpu`       | Número de GPUs           | 1      | 1 (múltiplas GPUs em dev)        |

### Ajustando para sua GPU

**GPU com 4GB VRAM:**
```bash
python train.py --dataset cifar10 --model dcgan --batch-size 32 --img-size 64
```

**GPU com 8GB+ VRAM:**
```bash
python train.py --dataset cifar10 --model wgan-gp --batch-size 128 --img-size 128 --ngf 128
```

**Sem GPU (CPU):**
```bash
python train.py --dataset mnist --model dcgan --batch-size 16 --epochs 10 --workers 4
```

---

## 📂 Estrutura de Saída

Após o treinamento, os resultados são salvos em:

```
outputs/
└── <dataset>/
    └── <model>_<timestamp>/
        ├── config.json              # Configurações do treinamento
        ├── training.log             # Log detalhado
        ├── training_losses.png      # Gráfico de perdas
        ├── final_samples.png        # Amostras finais
        ├── samples/                 # Amostras por época
        │   ├── epoch_5.png
        │   ├── epoch_10.png
        │   └── ...
        └── checkpoints/             # Checkpoints do modelo
            ├── checkpoint_epoch_10.pth
            ├── checkpoint_epoch_20.pth
            ├── checkpoint_latest.pth
            └── ...
```

### Arquivo config.json
```json
{
    "dataset": "cifar10",
    "model": "dcgan",
    "epochs": 50,
    "batch_size": 128,
    "img_size": 64,
    "lr": 0.0002,
    "nc": 3,
    "nz": 100,
    "saved_at": "2024-01-15 14:30:00"
}
```

---

## 📝 Dicas e Boas Práticas

### 1. Escolha do Dataset
- **Iniciante:** Comece com MNIST ou Fashion-MNIST (mais fácil)
- **Intermediário:** CIFAR-10 (colorido, mais desafiador)
- **Avançado:** CelebA ou Custom (alta resolução)

### 2. Escolha do Modelo
- **Prototipagem rápida:** DCGAN
- **Qualidade superior:** WGAN-GP (mais lento, mas melhor)

### 3. Número de Épocas
- **MNIST/Fashion-MNIST:** 25-50 épocas
- **CIFAR-10:** 50-100 épocas
- **CelebA/Custom:** 100-200 épocas

### 4. Monitoramento
- Verifique as amostras a cada 5 épocas
- Se as imagens não melhorarem após 20 épocas, ajuste hiperparâmetros
- Perdas muito próximas de 0 podem indicar problemas

### 5. Evitando Mode Collapse
- Use WGAN-GP ao invés de DCGAN
- Reduza learning rate se ocorrer
- Aumente número de épocas

### 6. Melhorando Qualidade
- Aumente `--ngf` e `--ndf` (ex: 128 ao invés de 64)
- Use `--img-size 128` para maior resolução
- Treine por mais épocas
- Use mais dados (>10.000 imagens)

---

## 🐛 Troubleshooting

### Problema: "CUDA out of memory"
**Solução:**
```bash
python train.py --dataset cifar10 --model dcgan --batch-size 32
```
Reduza `--batch-size` ou `--img-size`

---

### Problema: "Dataset não encontrado"
**Solução:**
- Datasets MNIST, Fashion-MNIST, CIFAR-10: Download automático, aguarde
- CelebA/Custom: Organize manualmente em `./data/<dataset>/`

---

### Problema: Imagens borradas
**Solução:**
- Treine por mais épocas
- Use WGAN-GP ao invés de DCGAN
- Aumente `--ngf` e `--ndf`

---

### Problema: Mode collapse (imagens todas iguais)
**Solução:**
```bash
python train.py --dataset cifar10 --model wgan-gp --epochs 100
```
Use WGAN-GP que é mais estável

---

### Problema: Treinamento muito lento
**Solução:**
- Reduza `--img-size` (ex: 32 ao invés de 64)
- Reduza `--batch-size`
- Use DCGAN ao invés de WGAN-GP
- Verifique se GPU está sendo usada (veja logs)

---

## 🎯 Resultados Esperados

### MNIST (25 épocas)
- ✅ Dígitos reconhecíveis
- ⏱️ ~5-10 minutos (GPU)

### Fashion-MNIST (50 épocas)
- ✅ Roupas com formas definidas
- ⏱️ ~10-20 minutos (GPU)

### CIFAR-10 (50 épocas)
- ✅ Objetos coloridos com formas básicas
- ⏱️ ~30-60 minutos (GPU)

### CIFAR-10 WGAN-GP (100 épocas)
- ✅ Imagens coloridas de boa qualidade
- ⏱️ ~2-3 horas (GPU)

---

## 📚 Recursos Adicionais

### Papers Originais
- **DCGAN:** [Unsupervised Representation Learning with DCGANs](https://arxiv.org/abs/1511.06434)
- **WGAN-GP:** [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

### Arquivos do Projeto
- `train.py` - Script principal de treinamento
- `models.py` - Arquiteturas dos modelos
- `config.py` - Configurações de datasets e modelos
- `utils.py` - Funções auxiliares

---

## 🤝 Contribuindo

Para adicionar novos datasets ou modelos:
1. Adicione configuração em `config.py`
2. Implemente arquitetura em `models.py` (se novo modelo)
3. Atualize `TRAINING_GUIDE.md`

---

## 📄 Licença

Este projeto é open source. Sinta-se livre para usar e modificar!

---

**Dúvidas?** Abra uma issue no GitHub! 🚀
