# 🪟 Guia Específico para Windows

Este guia contém informações e soluções específicas para usuários do Windows.

## 📋 Índice

- [Início Rápido](#início-rápido)
- [Resolução de Problemas](#resolução-de-problemas)
  - [Fashion-MNIST Download](#fashion-mnist-download)
  - [Caminhos com Espaços](#caminhos-com-espaços)
- [Sincronização com Google Drive](#sincronização-com-google-drive)
- [Logs e Diagnóstico](#logs-e-diagnóstico)

## 🚀 Início Rápido

### 1. Instalação (Primeira Vez)

```batch
REM Execute como Administrador (opcional, mas recomendado)
INSTALAR.bat
```

### 2. Iniciar o Sistema

```batch
REM Execute o inicializador principal
INICIAR.bat
```

O `INICIAR.bat` irá:
- ✅ Verificar a instalação
- ✅ Baixar modelos se necessário
- ✅ Abrir menu interativo para geração de imagens

## 🔧 Resolução de Problemas

### Fashion-MNIST Download

**Problema:** Erros ao baixar o dataset Fashion-MNIST, como:
```
HTTPError: HTTP Error 503: Service Unavailable
ConnectionError: Failed to download Fashion-MNIST
```

**Solução:** Configure o `TORCH_HOME` para um caminho sem espaços

#### Opção 1: Variável de Ambiente Permanente

1. Abra o **Painel de Controle**
2. Vá em **Sistema e Segurança** → **Sistema**
3. Clique em **Configurações avançadas do sistema**
4. Clique em **Variáveis de Ambiente**
5. Em **Variáveis do usuário**, clique em **Novo**
6. Configure:
   - Nome da variável: `TORCH_HOME`
   - Valor da variável: `C:\torch_data` (ou outro caminho sem espaços)
7. Clique em **OK** para salvar
8. **Reinicie o prompt de comando** ou reinicie o computador

#### Opção 2: Configurar Temporariamente (por sessão)

Antes de executar o `INICIAR.bat`, execute no prompt:

```batch
set TORCH_HOME=C:\torch_data
```

#### Opção 3: Modificar INICIAR.bat

Adicione esta linha no início do `INICIAR.bat` (após `@echo off`):

```batch
REM Configurar diretório de cache do PyTorch
set TORCH_HOME=C:\torch_data
```

### Caminhos com Espaços

**Problema:** Alguns componentes do PyTorch/TorchVision podem ter problemas com caminhos que contêm espaços, como:
- `C:\Program Files\Python`
- `C:\Users\Seu Nome\Documents`

**Soluções:**

1. **Use caminhos sem espaços para TORCH_HOME:**
   ```batch
   set TORCH_HOME=C:\torch_data
   ```

2. **Instale o Python em um caminho sem espaços:**
   - ✅ Bom: `C:\Python311`
   - ✅ Bom: `C:\dev\python`
   - ❌ Evite: `C:\Program Files\Python311`

3. **Clone o repositório em um caminho sem espaços:**
   - ✅ Bom: `C:\projetos\projeto-geracao-imagem`
   - ❌ Evite: `C:\Users\Seu Nome\Meus Documentos\projeto`

## ☁️ Sincronização com Google Drive

Mantenha seus modelos e resultados sincronizados automaticamente com o Google Drive.

### Opção 1: Google Drive para Desktop (Recomendado)

**Vantagens:**
- ✅ Sincronização automática e contínua
- ✅ Interface gráfica simples
- ✅ Não bloqueia o treinamento
- ✅ Sincroniza em segundo plano

**Como configurar:**

1. **Instale o Google Drive para Desktop:**
   - Baixe em: https://www.google.com/drive/download/
   - Instale e faça login com sua conta Google

2. **Configure a pasta do projeto:**

   **Método A: Projeto dentro do Google Drive**
   ```batch
   REM Clone ou mova o projeto para dentro da pasta do Google Drive
   cd G:\Meu Drive\projetos
   git clone https://github.com/drakko2002/projeto-geracao-imagem.git
   cd projeto-geracao-imagem
   INSTALAR.bat
   INICIAR.bat
   ```

   **Método B: Sincronizar apenas a pasta outputs**
   ```batch
   REM 1. Crie um link simbólico da pasta outputs para o Google Drive
   mklink /D "outputs" "G:\Meu Drive\gan-outputs"
   
   REM 2. Ou copie manualmente após o treinamento
   xcopy outputs "G:\Meu Drive\gan-outputs" /E /I /Y
   ```

3. **Configurar exclusões (opcional):**
   - Para evitar sincronizar arquivos temporários
   - Clique direito no ícone do Google Drive → Preferências
   - Em "Pastas" → Configure para não sincronizar:
     - `venv/` (ambiente virtual - não precisa sincronizar)
     - `data/` (datasets grandes - opcional)

### Opção 2: rclone (Avançado)

**Vantagens:**
- ✅ Mais controle sobre o que sincronizar
- ✅ Pode agendar sincronizações
- ✅ Suporta múltiplos provedores de nuvem

**Como configurar:**

1. **Instale o rclone:**
   - Baixe em: https://rclone.org/downloads/
   - Extraia para `C:\rclone`
   - Adicione `C:\rclone` ao PATH do Windows

2. **Configure o Google Drive:**
   ```batch
   rclone config
   ```
   - Siga o assistente para configurar "gdrive" como remote
   - Autorize o acesso à sua conta Google

3. **Crie um script de sincronização:**

   Crie `sync_to_drive.bat`:
   ```batch
   @echo off
   REM Sincronizar checkpoints e outputs para Google Drive
   echo Sincronizando com Google Drive...
   
   REM Sincronizar apenas checkpoints (mais rápido)
   rclone copy "outputs" "gdrive:gan-outputs" --include "*.pth" --progress
   
   REM Ou sincronizar tudo (mais lento)
   REM rclone sync "outputs" "gdrive:gan-outputs" --progress
   
   echo.
   echo Sincronizacao concluida!
   pause
   ```

4. **Use após o treinamento:**
   ```batch
   REM Após treinar um modelo
   INICIAR.bat
   REM ... treinar modelo ...
   
   REM Sincronizar com Google Drive
   sync_to_drive.bat
   ```

### Opção 3: Script de Backup Manual

**Para backups ocasionais sem instalar ferramentas extras:**

Crie `backup_outputs.bat`:
```batch
@echo off
setlocal enabledelayedexpansion

REM Configurar destino (ajuste conforme necessário)
set DESTINO=G:\Meu Drive\gan-backup

REM Criar pasta com data/hora
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/: " %%a in ('time /t') do (set mytime=%%a%%b)
set BACKUP_DIR=%DESTINO%\backup_%mydate%_%mytime%

echo.
echo Criando backup em: %BACKUP_DIR%
echo.

REM Copiar outputs
xcopy outputs "%BACKUP_DIR%\outputs" /E /I /Y /Q

echo.
echo Backup concluido!
pause
```

### 📊 Comparação de Métodos

| Método | Automático | Velocidade | Complexidade | Recomendado para |
|--------|-----------|------------|--------------|------------------|
| **Google Drive Desktop** | ✅ Sim | ⚡ Rápido | 🟢 Fácil | Todos os usuários |
| **rclone** | ⚠️ Manual* | ⚡⚡ Muito rápido | 🟡 Médio | Usuários avançados |
| **Script Manual** | ❌ Não | 🐌 Lento | 🟢 Fácil | Backups ocasionais |

*\* Pode ser automatizado com Agendador de Tarefas do Windows*

### 🔄 Dicas de Sincronização

1. **Sincronize apenas checkpoints importantes:**
   ```batch
   REM Checkpoints .pth são grandes (50-150MB cada)
   REM Considere manter apenas checkpoint_latest.pth
   ```

2. **Não sincronize durante o treinamento:**
   - Pode deixar o treinamento mais lento
   - Sincronize após completar o treinamento

3. **Use compressão para compartilhar:**
   ```batch
   REM Comprimir antes de fazer upload
   tar -czf modelo_mnist.tar.gz outputs/mnist/dcgan_xxx/checkpoints/checkpoint_latest.pth
   ```

## 📝 Logs e Diagnóstico

### Verificar Logs

O `INICIAR.bat` cria automaticamente um arquivo de log:

```
iniciar.log
```

**Para ver o log:**
```batch
REM Abrir no Notepad
notepad iniciar.log

REM Ou ver as últimas linhas
powershell Get-Content iniciar.log -Tail 50
```

### O que Verificar nos Logs

- ✅ **Mensagens de erro:** Procure por "Error", "Failed", "❌"
- ✅ **Avisos:** Procure por "Warning", "⚠️"
- ✅ **Checkpoints encontrados:** Verifique se os modelos foram detectados
- ✅ **Comandos executados:** Veja quais scripts Python foram rodados

### Habilitar Logs Detalhados

Edite `INICIAR.bat` e substitua:
```batch
python comando.py
```

Por:
```batch
python -v comando.py >> iniciar.log 2>&1
```

## 🎯 Dicas Adicionais

### Melhorar Performance

1. **Use SSD para o projeto:**
   - Datasets e checkpoints se beneficiam de SSDs
   - Evite HDs externos USB 2.0

2. **Desabilite o antivírus temporariamente:**
   - Alguns antivírus podem deixar o Python mais lento
   - Adicione exceção para a pasta do projeto

3. **Feche aplicações pesadas:**
   - Navegadores com muitas abas
   - Outros programas que usam GPU

### Atalhos Úteis

Crie atalhos no Desktop para acesso rápido:

1. **Atalho para INICIAR.bat:**
   - Clique direito em `INICIAR.bat`
   - "Criar atalho"
   - Arraste para o Desktop
   - Renomeie para "Gerador de Imagens IA"

2. **Atalho para app_gui.py:**
   - Crie um arquivo `Abrir_GUI.bat`:
   ```batch
   @echo off
   cd /d "%~dp0"
   call venv\Scripts\activate.bat
   python app_gui.py
   pause
   ```

## ❓ Precisa de Ajuda?

- 🐛 **Bugs:** Abra uma [issue](https://github.com/drakko2002/projeto-geracao-imagem/issues)
- 💡 **Dúvidas:** Verifique o [README principal](README.md)
- 📧 **Suporte:** Entre em contato através do GitHub

---

**Última atualização:** 2024-12-05
