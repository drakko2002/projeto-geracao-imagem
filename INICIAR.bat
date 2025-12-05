@echo off
chcp 65001 >nul 2>&1
REM ══════════════════════════════════════════════════════════════
REM  INICIALIZADOR PRINCIPAL - 1 CLIQUE PARA TUDO
REM  Baixa modelos + Abre menu de geração
REM ══════════════════════════════════════════════════════════════
REM
REM  DICAS PARA WINDOWS:
REM  - Configure TORCH_HOME para evitar problemas com Fashion-MNIST
REM    Ex: set TORCH_HOME=C:\torch_data
REM  - Veja WINDOWS_README.md para mais informações
REM  - Logs salvos em: iniciar.log
REM ══════════════════════════════════════════════════════════════

title Gerador de Imagens IA
color 0B

REM Redirecionar saída para log (captura stdout e stderr)
REM NOTA: O log é sobrescrito a cada execução
echo [%date% %time%] Iniciando INICIAR.bat > iniciar.log 2>&1

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
    echo [%date% %time%] ERRO: venv nao encontrado >> iniciar.log 2>&1
    pause
    exit /b 1
)

echo [%date% %time%] venv encontrado, ativando... >> iniciar.log 2>&1

REM Ativar ambiente
call venv\Scripts\activate.bat

cls
echo.
echo ==============================================================
echo  GERADOR DE IMAGENS COM INTELIGENCIA ARTIFICIAL
echo  Iniciando sistema...
echo ==============================================================
echo.

REM Verificar se modelos existem
echo Verificando modelos...
echo [%date% %time%] Verificando modelos existentes... >> iniciar.log 2>&1
python download_models.py --check >> iniciar.log 2>&1

REM Verificar se pelo menos um modelo existe
python -c "from pathlib import Path; import sys; sys.exit(0 if any(Path('outputs').rglob('checkpoint_latest.pth')) else 1)" 2>>iniciar.log

if %errorlevel% neq 0 (
    cls
    echo.
    echo ==============================================================
    echo           BAIXANDO MODELOS PRE-TREINADOS...
    echo 
    echo  (Isso vai demorar alguns minutos na primeira vez)
    echo ==============================================================
    echo.
    echo [%date% %time%] Nenhum modelo encontrado, baixando... >> iniciar.log 2>&1
    
    REM Baixar todos os modelos disponíveis
    python download_models.py --all >> iniciar.log 2>&1
    
    if %errorlevel% neq 0 (
        echo.
        echo ══════════════════════════════════════════════════════════════
        echo   Alguns modelos nao puderam ser baixados                     
        echo   Continuando com modelos disponiveis...                      
        echo ══════════════════════════════════════════════════════════════
        echo.
        echo [%date% %time%] Aviso: Alguns modelos falharam ao baixar >> iniciar.log 2>&1
        timeout /t 3 >nul
    )
)

REM Abrir menu principal
cls
:menu
echo.
echo ══════════════════════════════════════════════════════════════
echo                                                             
echo           GERADOR DE IMAGENS COM IA - MENU PRINCIPAL         
echo                                                              
echo ══════════════════════════════════════════════════════════════
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
echo   4. 🎨 Abrir Interface Grafica (app_gui.py)
echo.
echo   5. 🔄 Baixar/Atualizar modelos
echo.
echo   6. ℹ️  Informacoes do sistema
echo.
echo   0. ❌ Sair
echo.
echo ══════════════════════════════════════════════════════════════
echo.

set /p choice="Digite sua escolha [0-6]: "

if "%choice%"=="0" goto :end
if "%choice%"=="1" goto :mnist
if "%choice%"=="2" goto :cifar10
if "%choice%"=="3" goto :fashion
if "%choice%"=="4" goto :app_gui
if "%choice%"=="5" goto :download
if "%choice%"=="6" goto :info

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
echo ══════════════════════════════════════════════════════════════
echo           GERAR NUMEROS (0-9) - MNIST                        
echo ══════════════════════════════════════════════════════════════
echo.
echo [%date% %time%] Procurando checkpoint MNIST... >> iniciar.log 2>&1

REM Descobrir checkpoint dinamicamente
for /f "delims=" %%i in ('python find_checkpoint.py mnist 2^>^&1') do set MNIST_CHECKPOINT=%%i
if defined MNIST_CHECKPOINT (
    echo [%date% %time%] Checkpoint encontrado: %MNIST_CHECKPOINT% >> iniciar.log 2>&1
) else (
    echo [%date% %time%] Nenhum checkpoint MNIST encontrado >> iniciar.log 2>&1
)

REM Verificar se modelo existe
if not defined MNIST_CHECKPOINT (
    echo ❌ Modelo MNIST nao encontrado!
    echo.
    echo [%date% %time%] Checkpoint MNIST nao encontrado >> iniciar.log 2>&1
    echo Deseja baixar agora? (S/N)
    set /p dl="Sua escolha: "
    if /i "%dl%"=="S" (
        echo.
        echo Baixando modelo MNIST...
        echo [%date% %time%] Baixando modelo MNIST... >> iniciar.log 2>&1
        python download_models.py --model mnist >> iniciar.log 2>&1
        if %errorlevel% neq 0 (
            echo ✗ Falha ao baixar modelo
            echo [%date% %time%] Falha ao baixar MNIST >> iniciar.log 2>&1
            pause
            cls
            goto :menu
        )
        REM Tentar encontrar o checkpoint novamente após o download
        for /f "delims=" %%i in ('python find_checkpoint.py mnist 2^>^&1') do set MNIST_CHECKPOINT=%%i
        if not defined MNIST_CHECKPOINT (
            echo ✗ Checkpoint nao encontrado apos download
            echo [%date% %time%] ERRO: Checkpoint nao encontrado apos download >> iniciar.log 2>&1
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
echo [%date% %time%] Gerando imagem MNIST com prompt: %prompt% >> iniciar.log 2>&1

python generate_interactive.py --checkpoint "%MNIST_CHECKPOINT%" --prompt "%prompt%" --no-interactive >> iniciar.log 2>&1

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
echo ═════════════════════════════════════════════════════════════
echo        GERAR ANIMAIS E VEICULOS - CIFAR-10                   
echo ══════════════════════════════════════════════════════════════
echo.
echo [%date% %time%] Procurando checkpoint CIFAR-10... >> iniciar.log 2>&1

REM Descobrir checkpoint dinamicamente
for /f "delims=" %%i in ('python find_checkpoint.py cifar10 2^>^&1') do set CIFAR10_CHECKPOINT=%%i
if defined CIFAR10_CHECKPOINT (
    echo [%date% %time%] Checkpoint encontrado: %CIFAR10_CHECKPOINT% >> iniciar.log 2>&1
) else (
    echo [%date% %time%] Nenhum checkpoint CIFAR-10 encontrado >> iniciar.log 2>&1
)

REM Verificar se modelo existe
if not defined CIFAR10_CHECKPOINT (
    echo ❌ Modelo CIFAR-10 nao encontrado!
    echo.
    echo [%date% %time%] Checkpoint CIFAR-10 nao encontrado >> iniciar.log 2>&1
    echo Deseja baixar agora? (S/N)
    set /p dl="Sua escolha: "
    if /i "%dl%"=="S" (
        echo.
        echo Baixando modelo CIFAR-10...
        echo [%date% %time%] Baixando modelo CIFAR-10... >> iniciar.log 2>&1
        python download_models.py --model cifar10 >> iniciar.log 2>&1
        if %errorlevel% neq 0 (
            echo ✗ Falha ao baixar modelo
            echo [%date% %time%] Falha ao baixar CIFAR-10 >> iniciar.log 2>&1
            pause
            cls
            goto :menu
        )
        REM Tentar encontrar o checkpoint novamente após o download
        for /f "delims=" %%i in ('python find_checkpoint.py cifar10 2^>^&1') do set CIFAR10_CHECKPOINT=%%i
        if not defined CIFAR10_CHECKPOINT (
            echo ✗ Checkpoint nao encontrado apos download
            echo [%date% %time%] ERRO: Checkpoint nao encontrado apos download >> iniciar.log 2>&1
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
echo [%date% %time%] Gerando imagem CIFAR-10 com prompt: %prompt% >> iniciar.log 2>&1

python generate_interactive.py --checkpoint "%CIFAR10_CHECKPOINT%" --prompt "%prompt%" --no-interactive >> iniciar.log 2>&1

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
echo ══════════════════════════════════════════════════════════════
echo        GERAR ROUPAS E ACESSORIOS - Fashion-MNIST             
echo ══════════════════════════════════════════════════════════════
echo.
echo [%date% %time%] Procurando checkpoint Fashion-MNIST... >> iniciar.log 2>&1

REM Descobrir checkpoint dinamicamente
for /f "delims=" %%i in ('python find_checkpoint.py fashion-mnist 2^>^&1') do set FASHION_CHECKPOINT=%%i
if defined FASHION_CHECKPOINT (
    echo [%date% %time%] Checkpoint encontrado: %FASHION_CHECKPOINT% >> iniciar.log 2>&1
) else (
    echo [%date% %time%] Nenhum checkpoint Fashion-MNIST encontrado >> iniciar.log 2>&1
)

REM Verificar se modelo existe
if not defined FASHION_CHECKPOINT (
    echo ❌ Modelo Fashion-MNIST nao encontrado!
    echo.
    echo [%date% %time%] Checkpoint Fashion-MNIST nao encontrado >> iniciar.log 2>&1
    echo Deseja baixar agora? (S/N)
    set /p dl="Sua escolha: "
    if /i "%dl%"=="S" (
        echo.
        echo Baixando modelo Fashion-MNIST...
        echo [%date% %time%] Baixando modelo Fashion-MNIST... >> iniciar.log 2>&1
        python download_models.py --model fashion-mnist >> iniciar.log 2>&1
        if %errorlevel% neq 0 (
            echo ✗ Falha ao baixar modelo
            echo [%date% %time%] Falha ao baixar Fashion-MNIST >> iniciar.log 2>&1
            pause
            cls
            goto :menu
        )
        REM Tentar encontrar o checkpoint novamente após o download
        for /f "delims=" %%i in ('python find_checkpoint.py fashion-mnist 2^>^&1') do set FASHION_CHECKPOINT=%%i
        if not defined FASHION_CHECKPOINT (
            echo ✗ Checkpoint nao encontrado apos download
            echo [%date% %time%] ERRO: Checkpoint nao encontrado apos download >> iniciar.log 2>&1
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
echo [%date% %time%] Gerando imagem Fashion-MNIST com prompt: %prompt% >> iniciar.log 2>&1

python generate_interactive.py --checkpoint "%FASHION_CHECKPOINT%" --prompt "%prompt%" --no-interactive >> iniciar.log 2>&1

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
REM  INTERFACE GRAFICA - APP_GUI.PY
REM ══════════════════════════════════════════════════════════════
:app_gui
cls
echo.
echo ══════════════════════════════════════════════════════════════
echo           ABRINDO INTERFACE GRAFICA                          
echo ══════════════════════════════════════════════════════════════
echo.
echo Iniciando app_gui.py...
echo.
echo [%date% %time%] Abrindo app_gui.py >> iniciar.log 2>&1

REM Abrir a GUI
python app_gui.py >> iniciar.log 2>&1

if %errorlevel% neq 0 (
    echo.
    echo ✗ Erro ao abrir interface grafica
    echo [%date% %time%] Erro ao abrir app_gui.py >> iniciar.log 2>&1
) else (
    echo.
    echo Interface grafica fechada
    echo [%date% %time%] app_gui.py fechado normalmente >> iniciar.log 2>&1
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
echo ══════════════════════════════════════════════════════════════
echo           BAIXAR/ATUALIZAR MODELOS                           
echo ══════════════════════════════════════════════════════════════
echo.
echo [%date% %time%] Menu de download de modelos >> iniciar.log 2>&1

python download_models.py --check >> iniciar.log 2>&1

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
if "%dlchoice%"=="1" (
    echo [%date% %time%] Baixando todos os modelos... >> iniciar.log 2>&1
    python download_models.py --all >> iniciar.log 2>&1
)
if "%dlchoice%"=="2" (
    echo [%date% %time%] Baixando modelo MNIST... >> iniciar.log 2>&1
    python download_models.py --model mnist >> iniciar.log 2>&1
)
if "%dlchoice%"=="3" (
    echo [%date% %time%] Baixando modelo CIFAR-10... >> iniciar.log 2>&1
    python download_models.py --model cifar10 >> iniciar.log 2>&1
)
if "%dlchoice%"=="4" (
    echo [%date% %time%] Baixando modelo Fashion-MNIST... >> iniciar.log 2>&1
    python download_models.py --model fashion-mnist >> iniciar.log 2>&1
)

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
echo ══════════════════════════════════════════════════════════════
echo          INFORMACOES DO SISTEMA                             
echo ══════════════════════════════════════════════════════════════
echo.
echo [%date% %time%] Exibindo informacoes do sistema >> iniciar.log 2>&1

echo Python:
python --version
python --version >> iniciar.log 2>&1
echo.

echo PyTorch:
python -c "import torch; print(f'  Versao: {torch.__version__}'); print(f'  CUDA: {\"Sim\" if torch.cuda.is_available() else \"Nao\"}')"
python -c "import torch; print(f'  Versao: {torch.__version__}'); print(f'  CUDA: {\"Sim\" if torch.cuda.is_available() else \"Nao\"}')" >> iniciar.log 2>&1
echo.

echo GPU:
python -c "import torch; print(f'  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Nenhuma GPU NVIDIA detectada (usando CPU)\"}')"
python -c "import torch; print(f'  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Nenhuma GPU NVIDIA detectada (usando CPU)\"}')" >> iniciar.log 2>&1
echo.

echo Modelos instalados:
python download_models.py --check >> iniciar.log 2>&1
echo.

pause
cls
goto :menu

:end
cls
echo.
echo ══════════════════════════════════════════════════════════════
echo                                                               
echo           Obrigado por usar o Gerador de Imagens IA!         
echo                                                               
echo ══════════════════════════════════════════════════════════════
echo.
echo [%date% %time%] Programa encerrado normalmente >> iniciar.log 2>&1
timeout /t 2 >nul
exit /b 0
