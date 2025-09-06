# Baseada em Ubuntu 20.04 para compatibilidade
FROM python:3.10-slim-bookworm

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema (git, etc)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório do app
WORKDIR /app

# Copia dependências
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do projeto
COPY . .

# Define variável para o token Hugging Face (será passada no docker run)
ENV HUGGINGFACE_HUB_TOKEN="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"

# Comando padrão: roda o app
CMD ["python", "app.py"]
