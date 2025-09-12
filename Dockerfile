# Baseada em Ubuntu 20.04 para compatibilidade
FROM python:3.10-slim-bookworm

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema (git, etc)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch numpy<2 diffusers[torch] transformers accelerate safetensors
RUN pip install --no-cache-dir flask

# Cria diretório do app
WORKDIR /app

# Copia dependências
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do projeto
COPY . .

# Define variável para o token Hugging Face (será passada no docker run)
ENV HUGGINGFACE_HUB_TOKEN=local.env.token

# Comando padrão: roda o app
CMD ["python", "app.py"]
#docker run -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN ia-projeto
#docker build -t ia-projeto .
#docker run -p 5000:5000 -e HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN ia-projeto
