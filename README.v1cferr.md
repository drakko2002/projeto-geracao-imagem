# IA Image Generator (Offline)

Projeto de faculdade para geração de imagens offline via IA (Stable Diffusion) com suporte completo a GPU NVIDIA.

## ✅ Status do Projeto

- ✅ Configurado e funcionando com GPU NVIDIA RTX 3050
- ✅ PyTorch 2.8.0 com CUDA 12.1
- ✅ Stable Diffusion 2.1 baixado e configurado
- ✅ Ambiente virtual Python isolado

## � Configuração do Token HuggingFace

Para baixar e usar o modelo Stable Diffusion, você precisa de um token do HuggingFace:

1. **Acesse**: <https://huggingface.co/settings/tokens>
2. **Crie** um novo token (pode ser "Read")
3. **Configure** no arquivo `.env`:

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite e adicione seu token
nano .env
```

**Conteúdo do `.env`:**

```bash
HUGGINGFACE_HUB_TOKEN=seu_token_aqui
```

## �🚀 Instalação e Uso

### Pré-requisitos

- Arch Linux (ou similar) com drivers NVIDIA instalados
- GPU NVIDIA com drivers funcionando
- Python 3.10+ instalado

### Setup Rápido

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate

# 2. Rodar o projeto 
./run.sh
```

### Setup Completo (primeira vez)

```bash
# 1. Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 2. Instalar PyTorch com CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar token HuggingFace
cp .env.example .env
# Edite o arquivo .env e adicione seu token do HuggingFace
# Obtenha em: https://huggingface.co/settings/tokens

# 5. Baixar modelo (primeira execução)
python download_model.py

# 6. Gerar imagem
python app.py
```

## 🎯 Performance

- **GPU**: NVIDIA GeForce RTX 3050
- **CUDA**: Versão 13.0
- **Geração**: ~23 segundos para 50 steps
- **Memória GPU**: ~2GB utilizada
- **Modelo**: Stable Diffusion 2.1 (fp16)

## 📁 Estrutura

```bash
projeto-geracao-imagem/
├── venv/                    # Ambiente virtual Python
├── model/                   # Modelo Stable Diffusion baixado
├── app.py                   # Script principal de geração
├── download_model.py        # Script para baixar modelo
├── run.sh                   # Script facilitador
├── requirements.txt         # Dependências atualizadas
├── .env                     # Variáveis de ambiente (seu token)
├── .env.example            # Exemplo de configuração
└── saida.png               # Imagem gerada
```

## 🔧 Personalização

Edite o prompt em `app.py` linha 23:

```python
image = pipe("um gato astronauta explorando marte").images[0]
```

## 🐛 Resolução de Problemas

### GPU não detectada

```bash
nvidia-smi  # Verificar se GPU funciona
python -c "import torch; print(torch.cuda.is_available())"
```

### Problemas de memória

- Já configurado com `attention_slicing` e `xformers`
- Para GPUs com menos de 6GB, considere reduzir resolução

### Token HuggingFace inválido

- Obtenha novo token em: <https://huggingface.co/settings/tokens>
- Edite o arquivo `.env` e atualize a variável `HUGGINGFACE_HUB_TOKEN`
- Use o arquivo `.env.example` como referência
