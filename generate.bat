@echo off
chcp 65001 >nul 2>&1
REM ══════════════════════════════════════════════════════════════
REM  Script para gerar imagens facilmente no Windows
REM  Projeto: Sistema de Geração de Imagens com GANs
REM ══════════════════════════════════════════════════════════════

REM Ativar ambiente virtual
if not exist "venv\Scripts\activate.bat" (
    echo [ERRO] Ambiente virtual nao encontrado!
    echo        Execute setup.bat primeiro
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:menu
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║          GERADOR DE IMAGENS - GANs em Alta Resolucao        ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo ══════════════════════════════════════════════════════════════
echo  MENU PRINCIPAL
echo ══════════════════════════════════════════════════════════════
echo.
echo   1. Gerar imagem com MNIST (digitos 0-9)
echo   2. Gerar imagem com CIFAR-10 (animais, veiculos)
echo   3. Gerar imagem com Fashion-MNIST (roupas)
echo   4. Modo avancado (linha de comando customizada)
echo   5. Baixar modelos pre-treinados
echo   6. Verificar modelos disponiveis
echo   0. Sair
echo.
echo ══════════════════════════════════════════════════════════════
echo.

set /p choice="Escolha uma opcao [0-6]: "

if "%choice%"=="0" goto :end
if "%choice%"=="1" goto :mnist
if "%choice%"=="2" goto :cifar10
if "%choice%"=="3" goto :fashion
if "%choice%"=="4" goto :advanced
if "%choice%"=="5" goto :download
if "%choice%"=="6" goto :check

echo Opcao invalida!
timeout /t 2 >nul
goto :menu

:mnist
echo.
echo ══════════════════════════════════════════════════════════════
echo  GERAR COM MNIST (Digitos 0-9)
echo ══════════════════════════════════════════════════════════════
echo.
set /p prompt="Digite o que gerar (ex: numero 5): "

if "%prompt%"=="" (
    echo Prompt vazio! Usando interativo...
    python generate_interactive.py --checkpoint outputs/mnist/dcgan_pretrained/checkpoints/checkpoint_latest.pth
) else (
    python generate_interactive.py --checkpoint outputs/mnist/dcgan_pretrained/checkpoints/checkpoint_latest.pth --prompt "%prompt%"
)

echo.
pause
goto :menu

:cifar10
echo.
echo ══════════════════════════════════════════════════════════════
echo  GERAR COM CIFAR-10 (Animais e Veiculos)
echo ══════════════════════════════════════════════════════════════
echo.
echo Classes disponiveis:
echo   Animais: gato, cachorro, passaro, cavalo, cervo, sapo
echo   Veiculos: aviao, carro, navio, caminhao
echo.
set /p prompt="Digite o que gerar (ex: gerar um gato): "

if "%prompt%"=="" (
    echo Prompt vazio! Usando interativo...
    python generate_interactive.py --checkpoint outputs/cifar10/dcgan_pretrained/checkpoints/checkpoint_latest.pth
) else (
    python generate_interactive.py --checkpoint outputs/cifar10/dcgan_pretrained/checkpoints/checkpoint_latest.pth --prompt "%prompt%"
)

echo.
pause
goto :menu

:fashion
echo.
echo ══════════════════════════════════════════════════════════════
echo  GERAR COM FASHION-MNIST (Roupas e Acessorios)
echo ══════════════════════════════════════════════════════════════
echo.
echo Classes disponiveis:
echo   camiseta, calca, pullover, vestido, casaco
echo   sandalia, camisa, tenis, bolsa, bota
echo.
set /p prompt="Digite o que gerar (ex: camiseta): "

if "%prompt%"=="" (
    echo Prompt vazio! Usando interativo...
    python generate_interactive.py --checkpoint outputs/fashion-mnist/dcgan_pretrained/checkpoints/checkpoint_latest.pth
) else (
    python generate_interactive.py --checkpoint outputs/fashion-mnist/dcgan_pretrained/checkpoints/checkpoint_latest.pth --prompt "%prompt%"
)

echo.
pause
goto :menu

:advanced
echo.
echo ══════════════════════════════════════════════════════════════
echo  MODO AVANCADO
echo ══════════════════════════════════════════════════════════════
echo.
echo Parametros disponiveis:
echo   --prompt "texto"         Descrever o que gerar
echo   --class-name "classe"    Nome exato da classe
echo   --num-samples N          Numero de imagens (padrao: 1)
echo   --upscale N              Fator de upscaling (padrao: 8)
echo   --upscale-method METHOD  lanczos, bicubic, nearest
echo   --sharpen N              Nitidez 1.0-2.0 (padrao: 1.6)
echo   --device cuda/cpu        Dispositivo
echo.
echo Exemplo:
echo   python generate_interactive.py --checkpoint outputs/cifar10/.../checkpoint_latest.pth --prompt "gato" --upscale 16
echo.
echo.
set /p cmd="Digite o comando completo: "

if not "%cmd%"=="" (
    %cmd%
) else (
    echo Comando vazio!
)

echo.
pause
goto :menu

:download
echo.
echo ══════════════════════════════════════════════════════════════
echo  BAIXAR MODELOS PRE-TREINADOS
echo ══════════════════════════════════════════════════════════════
echo.
echo   1. Baixar todos os modelos (~150MB)
echo   2. Baixar apenas MNIST (~50MB)
echo   3. Baixar apenas CIFAR-10 (~80MB)
echo   4. Baixar apenas Fashion-MNIST (~50MB)
echo   0. Voltar
echo.
set /p dl_choice="Escolha [0-4]: "

if "%dl_choice%"=="0" goto :menu
if "%dl_choice%"=="1" python download_models.py --all
if "%dl_choice%"=="2" python download_models.py --model mnist
if "%dl_choice%"=="3" python download_models.py --model cifar10
if "%dl_choice%"=="4" python download_models.py --model fashion-mnist

echo.
pause
goto :menu

:check
echo.
echo ══════════════════════════════════════════════════════════════
echo  VERIFICAR MODELOS DISPONIVEIS
echo ══════════════════════════════════════════════════════════════
echo.
python download_models.py --check
echo.
pause
goto :menu

:end
echo.
echo Ate logo!
echo.
timeout /t 2 >nul
exit /b 0
