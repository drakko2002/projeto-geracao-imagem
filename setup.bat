@echo off
chcp 65001 >nul 2>&1
REM ══════════════════════════════════════════════════════════════
REM  Script de configuração rápida para Windows
REM  Projeto: Sistema de Geração de Imagens com GANs
REM ══════════════════════════════════════════════════════════════

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║          SETUP - Sistema de Geracao de Imagens GAN          ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado!
    echo.
    echo Por favor, instale Python 3.10+ de:
    echo https://www.python.org/downloads/
    echo.
    echo Certifique-se de marcar "Add Python to PATH" durante instalacao
    pause
    exit /b 1
)

echo [OK] Python encontrado
python --version
echo.

REM Verificar se já existe ambiente virtual
if exist "venv\Scripts\activate.bat" (
    echo [OK] Ambiente virtual ja existe
) else (
    echo [INFO] Criando ambiente virtual...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao criar ambiente virtual
        pause
        exit /b 1
    )
    echo [OK] Ambiente virtual criado
)
echo.

REM Ativar ambiente virtual
echo [INFO] Ativando ambiente virtual...
call venv\Scripts\activate.bat
echo.

REM Verificar se requirements.txt existe
if not exist "requirements.txt" (
    echo [ERRO] requirements.txt nao encontrado!
    pause
    exit /b 1
)

REM Perguntar sobre instalação de dependências
echo ══════════════════════════════════════════════════════════════
echo  INSTALACAO DE DEPENDENCIAS
echo ══════════════════════════════════════════════════════════════
echo.
echo Voce tem GPU NVIDIA (RTX 2050, etc)?
echo.
echo   1. SIM - Instalar PyTorch com CUDA (recomendado para GPU)
echo   2. NAO - Instalar PyTorch apenas CPU
echo.
set /p gpu_choice="Escolha [1/2]: "

if "%gpu_choice%"=="1" (
    echo.
    echo [INFO] Instalando PyTorch com suporte CUDA...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo.
    echo [INFO] Instalando PyTorch CPU...
    pip install torch torchvision torchaudio
)

if %errorlevel% neq 0 (
    echo [ERRO] Falha ao instalar PyTorch
    pause
    exit /b 1
)

echo.
echo [INFO] Instalando outras dependencias...
pip install -r requirements.txt
pip install gdown

if %errorlevel% neq 0 (
    echo [ERRO] Falha ao instalar dependencias
    pause
    exit /b 1
)

echo.
echo [OK] Dependencias instaladas com sucesso!
echo.

REM Verificar se CUDA está disponível
echo ══════════════════════════════════════════════════════════════
echo  VERIFICACAO DE GPU
echo ══════════════════════════════════════════════════════════════
echo.
python -c "import torch; print(f'CUDA disponivel: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Nenhuma (usando CPU)\"}')"
echo.

REM Verificar se modelos já existem
echo ══════════════════════════════════════════════════════════════
echo  VERIFICACAO DE MODELOS
echo ══════════════════════════════════════════════════════════════
echo.
python download_models.py --check
echo.

REM Perguntar sobre download de modelos
echo Deseja baixar modelos pre-treinados agora?
echo.
echo   1. SIM - Baixar todos os modelos (~150MB)
echo   2. NAO - Pular por enquanto
echo.
set /p download_choice="Escolha [1/2]: "

if "%download_choice%"=="1" (
    echo.
    echo [INFO] Baixando modelos pre-treinados...
    python download_models.py --all
    
    if %errorlevel% neq 0 (
        echo.
        echo [AVISO] Alguns modelos nao puderam ser baixados
        echo         Verifique se os IDs do Google Drive estao configurados
        echo         em download_models.py
    )
)

echo.
echo ══════════════════════════════════════════════════════════════
echo  SETUP CONCLUIDO!
echo ══════════════════════════════════════════════════════════════
echo.
echo Proximo passo:
echo   1. Certifique-se de que modelos estao baixados
echo   2. Execute: generate.bat
echo.
echo Ou use linha de comando:
echo   python generate_interactive.py --checkpoint outputs/mnist/dcgan_pretrained/checkpoints/checkpoint_latest.pth --prompt "numero 5"
echo.
pause
