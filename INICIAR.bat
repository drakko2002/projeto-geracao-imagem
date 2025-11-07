@echo off
chcp 65001 >nul 2>&1
REM ══════════════════════════════════════════════════════════════
REM  INICIALIZADOR PRINCIPAL - 1 CLIQUE PARA TUDO
REM  Baixa modelos + Abre menu de geração
REM ══════════════════════════════════════════════════════════════

title Gerador de Imagens IA
color 0B

REM Verificar se instalação foi feita
if not exist "venv\Scripts\activate.bat" (
    color 0E
    echo.
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║  ATENCAO: Sistema nao instalado!                             ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo Por favor, execute primeiro: INSTALAR.bat
    echo.
    pause
    exit /b 1
)

REM Ativar ambiente
call venv\Scripts\activate.bat

cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║          GERADOR DE IMAGENS COM INTELIGENCIA ARTIFICIAL     ║
echo ║                                                              ║
echo ║              Iniciando sistema...                           ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Verificar se modelos existem
echo Verificando modelos...
python download_models.py --check >nul 2>&1

REM Verificar se pelo menos um modelo existe
python -c "from pathlib import Path; import sys; sys.exit(0 if any(Path('outputs').rglob('checkpoint_latest.pth')) else 1)" 2>nul

if %errorlevel% neq 0 (
    cls
    echo.
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║                                                              ║
    echo ║          BAIXANDO MODELOS PRE-TREINADOS...                  ║
    echo ║                                                              ║
    echo ║  (Isso vai demorar alguns minutos na primeira vez)          ║
    echo ║                                                              ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo.
    
    REM Baixar todos os modelos disponíveis
    python download_models.py --all
    
    if %errorlevel% neq 0 (
        echo.
        echo ╔══════════════════════════════════════════════════════════════╗
        echo ║  Alguns modelos nao puderam ser baixados                     ║
        echo ║  Continuando com modelos disponiveis...                      ║
        echo ╚══════════════════════════════════════════════════════════════╝
        echo.
        timeout /t 3 >nul
    )
)

REM Abrir menu principal
cls
:menu
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║          GERADOR DE IMAGENS COM IA - MENU PRINCIPAL         ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo ══════════════════════════════════════════════════════════════
echo  ESCOLHA O QUE GERAR:
echo ══════════════════════════════════════════════════════════════
echo.
echo   1. 🔢 Numeros (0-9) - MNIST
echo.
echo   2. 🐱 Animais e Veiculos - CIFAR-10
echo      (gatos, cachorros, passaros, cavalos, avioes, carros...)
echo.
echo   3. 👕 Roupas e Acessorios - Fashion-MNIST
echo      (camisetas, calcas, bolsas, sapatos...)
echo.
echo   4. 🔄 Baixar/Atualizar modelos
echo.
echo   5. ℹ️  Informacoes do sistema
echo.
echo   0. ❌ Sair
echo.
echo ══════════════════════════════════════════════════════════════
echo.

set /p choice="Digite sua escolha [0-5]: "

if "%choice%"=="0" goto :end
if "%choice%"=="1" goto :mnist
if "%choice%"=="2" goto :cifar10
if "%choice%"=="3" goto :fashion
if "%choice%"=="4" goto :download
if "%choice%"=="5" goto :info

echo.
echo Opcao invalida!
timeout /t 2 >nul
cls
goto :menu

REM ══════════════════════════════════════════════════════════════
REM  MNIST - NUMEROS
REM ══════════════════════════════════════════════════════════════
:mnist
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║          GERAR NUMEROS (0-9) - MNIST                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Verificar se modelo existe
if not exist "outputs\mnist\dcgan_pretrained\checkpoints\checkpoint_latest.pth" (
    echo ❌ Modelo MNIST nao encontrado!
    echo.
    echo Deseja baixar agora? (S/N)
    set /p dl="Sua escolha: "
    if /i "%dl%"=="S" (
        echo.
        echo Baixando modelo MNIST...
        python download_models.py --model mnist
        if %errorlevel% neq 0 (
            echo ✗ Falha ao baixar modelo
            pause
            cls
            goto :menu
        )
    ) else (
        cls
        goto :menu
    )
)

echo Exemplos: "numero 5", "mostrar 7", "digito 0"
echo.
set /p prompt="O que voce quer gerar? "

if "%prompt%"=="" (
    set prompt=numero aleatorio
)

echo.
echo ══════════════════════════════════════════════════════════════
echo  GERANDO IMAGEM...
echo ══════════════════════════════════════════════════════════════
echo.

python generate_interactive.py --checkpoint "outputs\mnist\dcgan_pretrained\checkpoints\checkpoint_latest.pth" --prompt "%prompt%" --no-interactive

if %errorlevel% equ 0 (
    echo.
    echo ══════════════════════════════════════════════════════════════
    echo  ✅ IMAGEM GERADA COM SUCESSO!
    echo ══════════════════════════════════════════════════════════════
    echo.
    echo A imagem foi salva em: outputs\mnist\
    echo.
    
    REM Encontrar última imagem gerada
    for /f "delims=" %%i in ('dir /b /od "outputs\mnist\generated_*.png" 2^>nul') do set "lastimg=%%i"
    
    if defined lastimg (
        echo Abrindo imagem...
        start "" "outputs\mnist\%lastimg%"
    )
) else (
    echo.
    echo ✗ Erro ao gerar imagem
)

echo.
pause
cls
goto :menu

REM ══════════════════════════════════════════════════════════════
REM  CIFAR-10 - ANIMAIS E VEICULOS
REM ══════════════════════════════════════════════════════════════
:cifar10
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║       GERAR ANIMAIS E VEICULOS - CIFAR-10                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Verificar se modelo existe
if not exist "outputs\cifar10\dcgan_pretrained\checkpoints\checkpoint_latest.pth" (
    echo ❌ Modelo CIFAR-10 nao encontrado!
    echo.
    echo Deseja baixar agora? (S/N)
    set /p dl="Sua escolha: "
    if /i "%dl%"=="S" (
        echo.
        echo Baixando modelo CIFAR-10...
        python download_models.py --model cifar10
        if %errorlevel% neq 0 (
            echo ✗ Falha ao baixar modelo
            pause
            cls
            goto :menu
        )
    ) else (
        cls
        goto :menu
    )
)

echo Animais: gato, cachorro, passaro, cavalo, cervo, sapo
echo Veiculos: aviao, carro, navio, caminhao
echo.
echo Exemplos: "gerar um gato", "quero ver avioes", "cachorro"
echo.
set /p prompt="O que voce quer gerar? "

if "%prompt%"=="" (
    set prompt=imagem aleatoria
)

echo.
echo ══════════════════════════════════════════════════════════════
echo  GERANDO IMAGEM...
echo ══════════════════════════════════════════════════════════════
echo.

python generate_interactive.py --checkpoint "outputs\cifar10\dcgan_pretrained\checkpoints\checkpoint_latest.pth" --prompt "%prompt%" --no-interactive

if %errorlevel% equ 0 (
    echo.
    echo ══════════════════════════════════════════════════════════════
    echo  ✅ IMAGEM GERADA COM SUCESSO!
    echo ══════════════════════════════════════════════════════════════
    echo.
    echo A imagem foi salva em: outputs\cifar10\
    echo.
    
    REM Encontrar última imagem gerada
    for /f "delims=" %%i in ('dir /b /od "outputs\cifar10\generated_*.png" 2^>nul') do set "lastimg=%%i"
    
    if defined lastimg (
        echo Abrindo imagem...
        start "" "outputs\cifar10\%lastimg%"
    )
) else (
    echo.
    echo ✗ Erro ao gerar imagem
)

echo.
pause
cls
goto :menu

REM ══════════════════════════════════════════════════════════════
REM  FASHION-MNIST - ROUPAS
REM ══════════════════════════════════════════════════════════════
:fashion
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║       GERAR ROUPAS E ACESSORIOS - Fashion-MNIST             ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Verificar se modelo existe
if not exist "outputs\fashion-mnist\dcgan_pretrained\checkpoints\checkpoint_latest.pth" (
    echo ❌ Modelo Fashion-MNIST nao encontrado!
    echo.
    echo Deseja baixar agora? (S/N)
    set /p dl="Sua escolha: "
    if /i "%dl%"=="S" (
        echo.
        echo Baixando modelo Fashion-MNIST...
        python download_models.py --model fashion-mnist
        if %errorlevel% neq 0 (
            echo ✗ Falha ao baixar modelo
            pause
            cls
            goto :menu
        )
    ) else (
        cls
        goto :menu
    )
)

echo Disponiveis: camiseta, calca, pullover, vestido, casaco
echo              sandalia, camisa, tenis, bolsa, bota
echo.
echo Exemplos: "camiseta", "quero ver sapatos", "bolsa"
echo.
set /p prompt="O que voce quer gerar? "

if "%prompt%"=="" (
    set prompt=roupa aleatoria
)

echo.
echo ══════════════════════════════════════════════════════════════
echo  GERANDO IMAGEM...
echo ══════════════════════════════════════════════════════════════
echo.

python generate_interactive.py --checkpoint "outputs\fashion-mnist\dcgan_pretrained\checkpoints\checkpoint_latest.pth" --prompt "%prompt%" --no-interactive

if %errorlevel% equ 0 (
    echo.
    echo ══════════════════════════════════════════════════════════════
    echo  ✅ IMAGEM GERADA COM SUCESSO!
    echo ══════════════════════════════════════════════════════════════
    echo.
    echo A imagem foi salva em: outputs\fashion-mnist\
    echo.
    
    REM Encontrar última imagem gerada
    for /f "delims=" %%i in ('dir /b /od "outputs\fashion-mnist\generated_*.png" 2^>nul') do set "lastimg=%%i"
    
    if defined lastimg (
        echo Abrindo imagem...
        start "" "outputs\fashion-mnist\%lastimg%"
    )
) else (
    echo.
    echo ✗ Erro ao gerar imagem
)

echo.
pause
cls
goto :menu

REM ══════════════════════════════════════════════════════════════
REM  DOWNLOAD DE MODELOS
REM ══════════════════════════════════════════════════════════════
:download
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║          BAIXAR/ATUALIZAR MODELOS                           ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

python download_models.py --check

echo.
echo ══════════════════════════════════════════════════════════════
echo.
echo   1. Baixar TODOS os modelos
echo   2. Baixar apenas MNIST (numeros)
echo   3. Baixar apenas CIFAR-10 (animais/veiculos)
echo   4. Baixar apenas Fashion-MNIST (roupas)
echo   0. Voltar
echo.

set /p dlchoice="Sua escolha: "

if "%dlchoice%"=="0" (
    cls
    goto :menu
)
if "%dlchoice%"=="1" python download_models.py --all
if "%dlchoice%"=="2" python download_models.py --model mnist
if "%dlchoice%"=="3" python download_models.py --model cifar10
if "%dlchoice%"=="4" python download_models.py --model fashion-mnist

echo.
pause
cls
goto :menu

REM ══════════════════════════════════════════════════════════════
REM  INFORMAÇÕES DO SISTEMA
REM ══════════════════════════════════════════════════════════════
:info
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║          INFORMACOES DO SISTEMA                             ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo Python:
python --version
echo.

echo PyTorch:
python -c "import torch; print(f'  Versao: {torch.__version__}'); print(f'  CUDA: {\"Sim\" if torch.cuda.is_available() else \"Nao\"}')"
echo.

echo GPU:
python -c "import torch; print(f'  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Nenhuma GPU NVIDIA detectada (usando CPU)\"}')"
echo.

echo Modelos instalados:
python download_models.py --check
echo.

pause
cls
goto :menu

:end
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║          Obrigado por usar o Gerador de Imagens IA!         ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
timeout /t 2 >nul
exit /b 0
