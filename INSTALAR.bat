@echo off
REM ══════════════════════════════════════════════════════════════
REM  INSTALADOR AUTOMÁTICO - 1 CLIQUE
REM  Sistema de Geração de Imagens com GANs
REM ══════════════════════════════════════════════════════════════

title Instalador GAN - Geracao de Imagens
color 0A

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║          BEM-VINDO AO GERADOR DE IMAGENS COM IA!            ║
echo ║                                                              ║
echo ║     Este instalador vai configurar tudo automaticamente     ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo.
echo [1/5] Verificando Python...
echo.

REM Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║  ERRO: Python nao encontrado!                                ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo Por favor, instale Python primeiro:
    echo.
    echo   1. Abra: https://www.python.org/downloads/
    echo   2. Baixe Python 3.10 ou superior
    echo   3. Durante instalacao, MARQUE "Add Python to PATH"
    echo   4. Execute este instalador novamente
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo    ✓ Python encontrado: %PYTHON_VERSION%
echo.

echo [2/5] Criando ambiente virtual...
echo.

if exist "venv" (
    echo    ✓ Ambiente virtual ja existe
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        color 0C
        echo    ✗ Erro ao criar ambiente virtual
        pause
        exit /b 1
    )
    echo    ✓ Ambiente virtual criado
)
echo.

echo [3/5] Ativando ambiente...
echo.
call venv\Scripts\activate.bat
echo    ✓ Ambiente ativado
echo.

echo [4/5] Instalando dependencias...
echo.
echo    Detectando GPU NVIDIA...

nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✓ GPU NVIDIA detectada! Instalando PyTorch com CUDA...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
) else (
    echo    ! GPU nao detectada. Instalando PyTorch CPU...
    pip install torch torchvision torchaudio --quiet
)

echo    Instalando outras bibliotecas...
pip install -r requirements.txt --quiet
pip install gdown Pillow --quiet

if %errorlevel% neq 0 (
    color 0C
    echo    ✗ Erro ao instalar dependencias
    pause
    exit /b 1
)
echo    ✓ Dependencias instaladas
echo.

echo [5/5] Verificando GPU...
echo.
python -c "import torch; print('    GPU: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (sem CUDA)'))"
echo.

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║          ✓ INSTALACAO CONCLUIDA COM SUCESSO!                ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo.
echo Proximo passo:
echo   → Execute: INICIAR.bat
echo.
echo Isso vai baixar os modelos e abrir o menu de geracao!
echo.
pause
