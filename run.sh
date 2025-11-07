#!/bin/bash

# ╔══════════════════════════════════════════════════════════════╗
# ║                                                              ║
# ║     🎨 SISTEMA DE TREINAMENTO DE GANs - MENU UNIFICADO 🎨    ║
# ║                                                              ║
# ║  Script completo para treinar, gerar e gerenciar modelos GAN ║
# ║                                                              ║
# ╚══════════════════════════════════════════════════════════════╝

set -e

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÕES E CORES
# ═══════════════════════════════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# ═══════════════════════════════════════════════════════════════
# FUNÇÕES DE INTERFACE
# ═══════════════════════════════════════════════════════════════

show_banner() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║          🎨  SISTEMA DE TREINAMENTO DE GANs  🎨              ║"
    echo "║                                                              ║"
    echo "║     Treine e gere imagens com Deep Learning facilmente!     ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_main_menu() {
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  MENU PRINCIPAL${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} 🚀 Treinar novo modelo         ${CYAN}(assistente passo-a-passo)${NC}"
    echo -e "  ${GREEN}2)${NC} ⚡ Exemplos rápidos             ${CYAN}(treinamentos predefinidos)${NC}"
    echo -e "  ${GREEN}3)${NC} 🎨 Gerar imagens               ${CYAN}(aleatórias de modelos treinados)${NC}"
    echo -e "  ${GREEN}4)${NC} 🎯 Gerar por classe            ${CYAN}(escolha o que gerar!)${NC} ${YELLOW}← NOVO!${NC}"
    echo -e "  ${GREEN}5)${NC} � Upscale de imagens          ${CYAN}(aumentar resolução!)${NC} ${YELLOW}← NOVO!${NC}"
    echo -e "  ${GREEN}6)${NC} �📊 Ver treinamentos            ${CYAN}(status e resultados)${NC}"
    echo -e "  ${GREEN}7)${NC} 📦 Datasets disponíveis        ${CYAN}(listar e info)${NC}"
    echo -e "  ${GREEN}8)${NC} 🤖 Modelos disponíveis         ${CYAN}(DCGAN, WGAN-GP)${NC}"
    echo -e "  ${GREEN}9)${NC} 📖 Ajuda                       ${CYAN}(guia e troubleshooting)${NC}"
    echo -e "  ${GREEN}0)${NC} ❌ Sair"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO: TREINAR MODELO (ASSISTENTE COMPLETO)
# ═══════════════════════════════════════════════════════════════

train_model() {
    show_banner
    echo -e "${BOLD}${PURPLE}🚀 ASSISTENTE DE TREINAMENTO${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # ─────────────────────────────────────────────────────────────
    # PASSO 1: Selecionar Dataset
    # ─────────────────────────────────────────────────────────────
    echo -e "${BOLD}PASSO 1/4: Selecione o dataset${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} CIFAR-10      ${CYAN}(60k imagens coloridas 32x32 - aviões, carros, etc)${NC}"
    echo -e "  ${GREEN}2)${NC} MNIST         ${CYAN}(70k dígitos 0-9 em grayscale 28x28)${NC} ${YELLOW}← Recomendado para teste${NC}"
    echo -e "  ${GREEN}3)${NC} Fashion-MNIST ${CYAN}(70k roupas em grayscale 28x28)${NC}"
    echo -e "  ${GREEN}4)${NC} CelebA        ${CYAN}(200k faces - requer download manual)${NC}"
    echo -e "  ${GREEN}5)${NC} Custom        ${CYAN}(suas próprias imagens)${NC}"
    echo ""
    read -p "$(echo -e ${YELLOW}Digite o número [1-5]: ${NC})" dataset_choice
    
    case $dataset_choice in
        1) DATASET="cifar10" ;;
        2) DATASET="mnist" ;;
        3) DATASET="fashion-mnist" ;;
        4) DATASET="celeba" ;;
        5) DATASET="custom" ;;
        *) echo -e "${RED}Opção inválida!${NC}"; sleep 2; return ;;
    esac
    
    echo -e "  ${GREEN}✓${NC} Dataset selecionado: ${CYAN}${DATASET}${NC}"
    echo ""
    
    # ─────────────────────────────────────────────────────────────
    # PASSO 2: Selecionar Modelo
    # ─────────────────────────────────────────────────────────────
    echo -e "${BOLD}PASSO 2/4: Selecione o modelo GAN${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} DCGAN    ${CYAN}(Rápido, estável, bom para iniciantes)${NC} ${YELLOW}← Recomendado${NC}"
    echo -e "  ${GREEN}2)${NC} WGAN-GP  ${CYAN}(Mais lento, melhor qualidade, mais estável)${NC}"
    echo ""
    read -p "$(echo -e ${YELLOW}Digite o número [1-2]: ${NC})" model_choice
    
    case $model_choice in
        1) MODEL="dcgan" ;;
        2) MODEL="wgan-gp" ;;
        *) echo -e "${RED}Opção inválida!${NC}"; sleep 2; return ;;
    esac
    
    echo -e "  ${GREEN}✓${NC} Modelo selecionado: ${CYAN}${MODEL}${NC}"
    echo ""
    
    # ─────────────────────────────────────────────────────────────
    # PASSO 3: Configurar Épocas
    # ─────────────────────────────────────────────────────────────
    echo -e "${BOLD}PASSO 3/4: Quantas épocas treinar?${NC}"
    echo ""
    echo -e "  ${CYAN}💡 Recomendações por dataset:${NC}"
    echo -e "     • MNIST: 25-50 épocas (~15-30 min)"
    echo -e "     • Fashion-MNIST: 50-75 épocas (~30-45 min)"
    echo -e "     • CIFAR-10: 50-100 épocas (~1-2 horas)"
    echo ""
    echo -e "  ${CYAN}💡 Para testes rápidos: 5-10 épocas${NC}"
    echo ""
    read -p "$(echo -e ${YELLOW}Número de épocas [padrão: 25]: ${NC})" epochs
    EPOCHS=${epochs:-25}
    
    echo -e "  ${GREEN}✓${NC} Épocas configuradas: ${CYAN}${EPOCHS}${NC}"
    echo ""
    
    # ─────────────────────────────────────────────────────────────
    # PASSO 4: Batch Size (Opcional/Avançado)
    # ─────────────────────────────────────────────────────────────
    echo -e "${BOLD}PASSO 4/4: Batch size (ENTER para usar padrão)${NC}"
    echo ""
    echo -e "  ${CYAN}💡 Recomendações por memória GPU:${NC}"
    echo -e "     • 16GB+ VRAM: 128-256"
    echo -e "     • 8GB VRAM: 64-128"
    echo -e "     • 4GB VRAM: 32-64"
    echo -e "     • CPU: 32"
    echo ""
    read -p "$(echo -e ${YELLOW}Batch size [padrão: 128]: ${NC})" batch_size
    BATCH_SIZE=${batch_size:-128}
    
    echo -e "  ${GREEN}✓${NC} Batch size configurado: ${CYAN}${BATCH_SIZE}${NC}"
    echo ""
    
    # ─────────────────────────────────────────────────────────────
    # RESUMO E CONFIRMAÇÃO
    # ─────────────────────────────────────────────────────────────
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${GREEN}📋 RESUMO DA CONFIGURAÇÃO:${NC}"
    echo ""
    echo -e "  Dataset:    ${CYAN}${DATASET}${NC}"
    echo -e "  Modelo:     ${CYAN}${MODEL}${NC}"
    echo -e "  Épocas:     ${CYAN}${EPOCHS}${NC}"
    echo -e "  Batch Size: ${CYAN}${BATCH_SIZE}${NC}"
    echo ""
    
    # Estimar tempo
    if [[ "$DATASET" == "mnist" ]]; then
        TIME_EST="~15-30 minutos"
    elif [[ "$DATASET" == "cifar10" ]]; then
        TIME_EST="~1-2 horas"
    else
        TIME_EST="variável"
    fi
    
    echo -e "  ${PURPLE}⏱️  Tempo estimado: ${TIME_EST}${NC}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    read -p "$(echo -e ${YELLOW}${BOLD}Confirmar e iniciar treinamento? [s/N]: ${NC})" confirm
    
    if [[ $confirm =~ ^[Ss]$ ]]; then
        echo ""
        echo -e "${GREEN}${BOLD}✨ Iniciando treinamento...${NC}"
        echo ""
        sleep 1
        
        # Executar treinamento
        python train.py \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE"
        
        echo ""
        echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}${BOLD}✅ TREINAMENTO CONCLUÍDO COM SUCESSO!${NC}"
        echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo -e "${CYAN}💡 Próximo passo: Gere imagens usando a opção 3 do menu!${NC}"
        echo ""
        read -p "Pressione Enter para voltar ao menu..."
    else
        echo -e "${YELLOW}Treinamento cancelado.${NC}"
        sleep 2
    fi
}

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO: EXEMPLOS RÁPIDOS
# ═══════════════════════════════════════════════════════════════

quick_examples() {
    show_banner
    echo -e "${BOLD}${PURPLE}⚡ EXEMPLOS RÁPIDOS DE TREINAMENTO${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${CYAN}Escolha um exemplo predefinido:${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} 🏃 Teste super rápido    ${YELLOW}(MNIST + DCGAN, 5 épocas, ~5 min)${NC}"
    echo -e "  ${GREEN}2)${NC} 🚀 Teste básico          ${YELLOW}(MNIST + DCGAN, 25 épocas, ~15 min)${NC}"
    echo -e "  ${GREEN}3)${NC} 🎨 Qualidade boa         ${YELLOW}(CIFAR-10 + DCGAN, 50 épocas, ~1h)${NC}"
    echo -e "  ${GREEN}4)${NC} ⭐ Alta qualidade        ${YELLOW}(CIFAR-10 + WGAN-GP, 100 épocas, ~3h)${NC}"
    echo -e "  ${GREEN}5)${NC} 👗 Fashion-MNIST         ${YELLOW}(Fashion + DCGAN, 50 épocas, ~30 min)${NC}"
    echo -e "  ${GREEN}0)${NC} ⬅️  Voltar"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    read -p "$(echo -e ${YELLOW}Escolha uma opção: ${NC})" choice
    
    case $choice in
        1)
            echo ""
            echo -e "${GREEN}🏃 Iniciando teste super rápido...${NC}"
            echo ""
            python train.py --dataset mnist --model dcgan --epochs 5 --batch-size 128
            ;;
        2)
            echo ""
            echo -e "${GREEN}🚀 Iniciando teste básico...${NC}"
            echo ""
            python train.py --dataset mnist --model dcgan --epochs 25 --batch-size 128
            ;;
        3)
            echo ""
            echo -e "${GREEN}🎨 Iniciando treinamento de qualidade boa...${NC}"
            echo ""
            python train.py --dataset cifar10 --model dcgan --epochs 50 --batch-size 128
            ;;
        4)
            echo ""
            echo -e "${GREEN}⭐ Iniciando treinamento de alta qualidade...${NC}"
            echo ""
            python train.py --dataset cifar10 --model wgan-gp --epochs 100 --batch-size 64
            ;;
        5)
            echo ""
            echo -e "${GREEN}👗 Iniciando treinamento Fashion-MNIST...${NC}"
            echo ""
            python train.py --dataset fashion-mnist --model dcgan --epochs 50 --batch-size 128
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}Opção inválida!${NC}"
            sleep 2
            return
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}${BOLD}✅ Treinamento concluído!${NC}"
    echo ""
    read -p "Pressione Enter para voltar ao menu..."
}

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO: GERAR IMAGENS
# ═══════════════════════════════════════════════════════════════

generate_images() {
    show_banner
    echo -e "${BOLD}${PURPLE}🎨 GERADOR DE IMAGENS${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${BOLD}Escolha o modo de geração:${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} Automático ${CYAN}(encontra último modelo treinado)${NC} ${YELLOW}← Recomendado${NC}"
    echo -e "  ${GREEN}2)${NC} Manual     ${CYAN}(especificar caminho do checkpoint)${NC}"
    echo -e "  ${GREEN}0)${NC} ⬅️  Voltar"
    echo ""
    read -p "$(echo -e ${YELLOW}Digite o número: ${NC})" gen_choice
    
    case $gen_choice in
        1)
            # Modo automático
            echo ""
            echo -e "${CYAN}🔍 Procurando modelos treinados...${NC}"
            echo ""
            python quick_generate.py
            ;;
        2)
            # Modo manual
            echo ""
            echo -e "${CYAN}📁 Checkpoints disponíveis:${NC}"
            echo ""
            find outputs -name "checkpoint_latest.pth" -type f 2>/dev/null | head -10
            echo ""
            read -p "$(echo -e ${YELLOW}Cole o caminho do checkpoint: ${NC})" checkpoint_path
            
            if [ ! -f "$checkpoint_path" ]; then
                echo -e "${RED}❌ Arquivo não encontrado!${NC}"
                sleep 2
                return
            fi
            
            read -p "$(echo -e ${YELLOW}Número de imagens [padrão: 64]: ${NC})" num_samples
            NUM_SAMPLES=${num_samples:-64}
            
            echo ""
            echo -e "${GREEN}🎨 Gerando $NUM_SAMPLES imagens...${NC}"
            echo ""
            
            python generate.py \
                --checkpoint "$checkpoint_path" \
                --num-samples "$NUM_SAMPLES"
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}Opção inválida!${NC}"
            sleep 2
            return
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}✅ Imagens geradas com sucesso!${NC}"
    echo ""
    read -p "Pressione Enter para voltar ao menu..."
}

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO: GERAR IMAGENS POR CLASSE (INTERATIVO)
# ═══════════════════════════════════════════════════════════════

generate_by_class() {
    show_banner
    echo -e "${BOLD}${PURPLE}🎯 GERADOR POR CLASSE/CATEGORIA${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${CYAN}Este modo permite escolher o que gerar:${NC}"
    echo ""
    echo -e "  • Gatos, cachorros, aviões (CIFAR-10)"
    echo -e "  • Dígitos específicos (MNIST)"
    echo -e "  • Roupas específicas (Fashion-MNIST)"
    echo -e "  • Usar prompts de texto!"
    echo ""
    echo -e "${YELLOW}⚠️  Nota: Para GANs incondicionais, a seleção é simulada.${NC}"
    echo -e "${YELLOW}    Para controle real, treine um Conditional GAN (c-GAN).${NC}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${BOLD}Escolha o checkpoint:${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} Usar último modelo treinado ${CYAN}(automático)${NC}"
    echo -e "  ${GREEN}2)${NC} Especificar checkpoint manualmente"
    echo -e "  ${GREEN}0)${NC} ⬅️  Voltar"
    echo ""
    read -p "$(echo -e ${YELLOW}Digite o número: ${NC})" checkpoint_choice
    
    CHECKPOINT_PATH=""
    
    case $checkpoint_choice in
        1)
            # Encontrar último checkpoint
            echo ""
            echo -e "${CYAN}🔍 Procurando último modelo...${NC}"
            CHECKPOINT_PATH=$(find outputs -name "checkpoint_latest.pth" -type f 2>/dev/null | head -1)
            
            if [ -z "$CHECKPOINT_PATH" ]; then
                echo -e "${RED}❌ Nenhum modelo encontrado!${NC}"
                echo -e "${YELLOW}Treine um modelo primeiro (opção 1 ou 2 do menu).${NC}"
                sleep 3
                return
            fi
            
            echo -e "${GREEN}✓ Encontrado: $CHECKPOINT_PATH${NC}"
            ;;
        2)
            # Manual
            echo ""
            echo -e "${CYAN}📁 Checkpoints disponíveis:${NC}"
            echo ""
            find outputs -name "checkpoint_latest.pth" -type f 2>/dev/null | head -10
            echo ""
            read -p "$(echo -e ${YELLOW}Cole o caminho do checkpoint: ${NC})" CHECKPOINT_PATH
            
            if [ ! -f "$CHECKPOINT_PATH" ]; then
                echo -e "${RED}❌ Arquivo não encontrado!${NC}"
                sleep 2
                return
            fi
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}Opção inválida!${NC}"
            sleep 2
            return
            ;;
    esac
    
    # Agora escolher modo de geração
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Modo de geração:${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} Modo interativo    ${CYAN}(menu com classes disponíveis)${NC} ${YELLOW}← Recomendado${NC}"
    echo -e "  ${GREEN}2)${NC} Prompt de texto    ${CYAN}(ex: 'gerar um gato')${NC}"
    echo -e "  ${GREEN}3)${NC} Classe específica  ${CYAN}(especificar diretamente)${NC}"
    echo -e "  ${GREEN}0)${NC} ⬅️  Voltar"
    echo ""
    read -p "$(echo -e ${YELLOW}Digite o número: ${NC})" mode_choice
    
    case $mode_choice in
        1)
            # Modo interativo (padrão do script) - 1 imagem em alta resolução
            echo ""
            echo -e "${GREEN}🎨 Modo interativo - Geração em alta resolução${NC}"
            echo -e "${CYAN}   • Gera 1 imagem por vez${NC}"
            echo -e "${CYAN}   • Upscaling automático 8x (ex: 28x28 → 224x224)${NC}"
            echo -e "${CYAN}   • Alta qualidade com nitidez aprimorada${NC}"
            echo ""
            python generate_interactive.py --checkpoint "$CHECKPOINT_PATH"
            ;;
        2)
            # Prompt de texto - 1 imagem em alta resolução
            echo ""
            echo -e "${CYAN}💬 Digite o que você quer gerar:${NC}"
            echo -e "${CYAN}   Exemplos:${NC}"
            echo -e "   • 'gerar um gato'"
            echo -e "   • 'quero ver aviões'"
            echo -e "   • 'mostrar o número 5'"
            echo ""
            read -p "$(echo -e ${YELLOW}Prompt: ${NC})" prompt
            
            if [ -z "$prompt" ]; then
                echo -e "${RED}❌ Prompt vazio!${NC}"
                sleep 2
                return
            fi
            
            echo ""
            echo -e "${GREEN}🎨 Gerando 1 imagem em alta resolução com prompt: '$prompt'${NC}"
            echo ""
            python generate_interactive.py \
                --checkpoint "$CHECKPOINT_PATH" \
                --prompt "$prompt" \
                --num-samples 1
            ;;
        3)
            # Classe específica - opção de múltiplas ou única
            echo ""
            read -p "$(echo -e ${YELLOW}"Nome da classe (ex: gato, 5, Camiseta): "${NC})" class_name
            
            if [ -z "$class_name" ]; then
                echo -e "${RED}❌ Nome vazio!${NC}"
                sleep 2
                return
            fi
            
            echo ""
            echo -e "${CYAN}Quantas imagens gerar?${NC}"
            echo -e "  ${GREEN}1)${NC} 1 imagem em alta resolução ${YELLOW}(Recomendado)${NC}"
            echo -e "  ${GREEN}2)${NC} Múltiplas imagens (grid)"
            echo ""
            read -p "$(echo -e ${YELLOW}Escolha [1-2]: ${NC})" img_mode
            
            case $img_mode in
                1|"")
                    NUM_SAMPLES=1
                    echo ""
                    echo -e "${GREEN}🎨 Gerando 1 imagem em alta resolução: '$class_name'${NC}"
                    ;;
                2)
                    read -p "$(echo -e ${YELLOW}Quantas imagens? [padrão: 16]: ${NC})" num_samples
                    NUM_SAMPLES=${num_samples:-16}
                    echo ""
                    echo -e "${GREEN}🎨 Gerando $NUM_SAMPLES imagens: '$class_name'${NC}"
                    ;;
                *)
                    NUM_SAMPLES=1
                    echo ""
                    echo -e "${GREEN}🎨 Gerando 1 imagem em alta resolução: '$class_name'${NC}"
                    ;;
            esac
            
            echo ""
            python generate_interactive.py \
                --checkpoint "$CHECKPOINT_PATH" \
                --class-name "$class_name" \
                --num-samples "$NUM_SAMPLES"
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}Opção inválida!${NC}"
            sleep 2
            return
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}✅ Geração concluída!${NC}"
    echo ""
    read -p "Pressione Enter para voltar ao menu..."
}

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO: UPSCALE DE IMAGENS
# ═══════════════════════════════════════════════════════════════

upscale_images() {
    show_banner
    echo -e "${BOLD}${PURPLE}📐 UPSCALE DE IMAGENS (AUMENTAR RESOLUÇÃO)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${CYAN}💡 Sobre Upscaling:${NC}"
    echo "   Aumenta a resolução de imagens geradas usando algoritmos avançados"
    echo "   Exemplo: 28x28 → 224x224 (8x maior)"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Listar imagens geradas disponíveis
    echo -e "${BOLD}Imagens geradas disponíveis:${NC}"
    echo ""
    
    IMAGES=()
    COUNT=0
    
    if [ -d "outputs" ]; then
        while IFS= read -r -d '' img; do
            COUNT=$((COUNT + 1))
            IMAGES+=("$img")
            SIZE=$(identify -format "%wx%h" "$img" 2>/dev/null || echo "desconhecido")
            FILESIZE=$(du -h "$img" | cut -f1)
            echo -e "  ${GREEN}$COUNT)${NC} $(basename "$img")"
            echo -e "      ${CYAN}Caminho: $img${NC}"
            echo -e "      ${CYAN}Tamanho: $SIZE | Arquivo: $FILESIZE${NC}"
            echo ""
        done < <(find outputs -name "*.png" -o -name "*.jpg" -print0 | sort -z)
    fi
    
    if [ $COUNT -eq 0 ]; then
        echo -e "${YELLOW}📭 Nenhuma imagem encontrada.${NC}"
        echo ""
        echo -e "${CYAN}💡 Gere imagens primeiro (opção 3 ou 4)!${NC}"
        echo ""
        read -p "Pressione Enter para voltar..."
        return
    fi
    
    echo -e "  ${GREEN}0)${NC} 🔙 Voltar ao menu"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    read -p "$(echo -e ${YELLOW}Escolha a imagem [0-$COUNT]: ${NC})" img_choice
    
    if [ "$img_choice" = "0" ]; then
        return
    fi
    
    if [ "$img_choice" -lt 1 ] || [ "$img_choice" -gt $COUNT ]; then
        echo -e "${RED}Opção inválida!${NC}"
        sleep 2
        return
    fi
    
    SELECTED_IMAGE="${IMAGES[$((img_choice - 1))]}"
    
    echo ""
    echo -e "${BOLD}Imagem selecionada:${NC} $(basename "$SELECTED_IMAGE")"
    echo ""
    
    # Escolher método de upscale
    echo -e "${BOLD}Escolha o método de upscaling:${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} Bicubic  ${CYAN}(rápido, boa qualidade)${NC} ${YELLOW}← Recomendado${NC}"
    echo -e "  ${GREEN}2)${NC} Lanczos  ${CYAN}(melhor qualidade, um pouco mais lento)${NC}"
    echo -e "  ${GREEN}3)${NC} Nearest  ${CYAN}(pixel-art, estilo retro)${NC}"
    echo -e "  ${GREEN}4)${NC} ESRGAN   ${CYAN}(super-resolução AI - requer instalação extra)${NC}"
    echo ""
    read -p "$(echo -e ${YELLOW}Método [1-4]: ${NC})" method_choice
    
    case $method_choice in
        1) METHOD="bicubic" ;;
        2) METHOD="lanczos" ;;
        3) METHOD="nearest" ;;
        4) METHOD="esrgan" ;;
        *) 
            echo -e "${RED}Opção inválida! Usando bicubic.${NC}"
            METHOD="bicubic"
            ;;
    esac
    
    # Escolher escala
    echo ""
    echo -e "${BOLD}Fator de escala:${NC}"
    echo ""
    echo -e "  ${CYAN}Sugestões por dataset:${NC}"
    echo -e "    MNIST/Fashion-MNIST (28x28):  ${GREEN}8x${NC} = 224x224 (web/redes sociais)"
    echo -e "    MNIST/Fashion-MNIST (28x28): ${GREEN}10x${NC} = 280x280 (Instagram)"
    echo -e "    CIFAR-10 (32x32):             ${GREEN}8x${NC} = 256x256 (web)"
    echo -e "    CIFAR-10 (32x32):            ${GREEN}16x${NC} = 512x512 (impressão)"
    echo ""
    read -p "$(echo -e ${YELLOW}Digite a escala [2-16]: ${NC})" scale
    
    # Validar escala
    if ! [[ "$scale" =~ ^[0-9]+$ ]] || [ "$scale" -lt 2 ] || [ "$scale" -gt 16 ]; then
        echo -e "${RED}Escala inválida! Usando 8x.${NC}"
        scale=8
    fi
    
    # Perguntar sobre melhorias
    echo ""
    echo -e "${BOLD}Aplicar melhorias de qualidade?${NC}"
    echo ""
    read -p "$(echo -e ${YELLOW}Aumentar nitidez? [s/N]: ${NC})" sharpen_choice
    read -p "$(echo -e ${YELLOW}Melhorar contraste? [s/N]: ${NC})" contrast_choice
    
    # Construir comando
    CMD="python scripts/upscale_images.py --input \"$SELECTED_IMAGE\" --scale $scale --method $METHOD"
    
    if [[ "$sharpen_choice" =~ ^[Ss]$ ]]; then
        CMD="$CMD --sharpen 1.6"
    fi
    
    if [[ "$contrast_choice" =~ ^[Ss]$ ]]; then
        CMD="$CMD --contrast 1.2"
    fi
    
    # Executar upscale
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🚀 Executando upscale...${NC}"
    echo ""
    echo -e "${CYAN}Comando: $CMD${NC}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    eval $CMD
    
    echo ""
    echo -e "${GREEN}✅ Upscale concluído!${NC}"
    echo ""
    echo -e "${CYAN}💡 Dica: O arquivo foi salvo com sufixo '_upscaled_${scale}x'${NC}"
    echo ""
    read -p "Pressione Enter para voltar ao menu..."
}

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO: VER STATUS DOS TREINAMENTOS
# ═══════════════════════════════════════════════════════════════

show_status() {
    show_banner
    echo -e "${BOLD}${PURPLE}📊 STATUS DE TREINAMENTOS${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    if [ ! -d "outputs" ] || [ -z "$(ls -A outputs 2>/dev/null)" ]; then
        echo -e "${YELLOW}📭 Nenhum treinamento encontrado.${NC}"
        echo ""
        echo -e "${CYAN}💡 Execute a opção 1 ou 2 para treinar seu primeiro modelo!${NC}"
    else
        echo -e "${BOLD}Treinamentos encontrados:${NC}"
        echo ""
        
        COUNTER=0
        
        # Listar diretórios de treinamento
        find outputs -type d -name "*_202*" 2>/dev/null | sort -r | while read -r dir; do
            if [ -f "$dir/config.json" ]; then
                COUNTER=$((COUNTER + 1))
                
                dataset=$(basename "$(dirname "$dir")")
                run=$(basename "$dir")
                
                echo -e "${GREEN}[$COUNTER]${NC} ${BOLD}$dataset${NC} - $run"
                echo -e "    📁 $dir"
                
                # Contar checkpoints
                checkpoint_count=$(find "$dir/checkpoints" -name "*.pth" 2>/dev/null | wc -l)
                echo -e "    💾 Checkpoints: ${PURPLE}$checkpoint_count${NC}"
                
                # Verificar status
                if [ -f "$dir/final_samples.png" ]; then
                    echo -e "    ✅ Status: ${GREEN}Concluído${NC}"
                else
                    echo -e "    ⏸️  Status: ${YELLOW}Em andamento${NC}"
                fi
                
                # Mostrar caminho do último checkpoint
                if [ -f "$dir/checkpoints/checkpoint_latest.pth" ]; then
                    echo -e "    🎯 Último checkpoint: ${CYAN}$dir/checkpoints/checkpoint_latest.pth${NC}"
                fi
                
                echo ""
            fi
        done
        
        if [ $COUNTER -eq 0 ]; then
            echo -e "${YELLOW}Nenhum treinamento válido encontrado.${NC}"
        fi
    fi
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    read -p "Pressione Enter para voltar ao menu..."
}

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO: LISTAR DATASETS
# ═══════════════════════════════════════════════════════════════

list_datasets() {
    show_banner
    echo -e "${BOLD}${PURPLE}📦 DATASETS DISPONÍVEIS${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    python train.py --list-datasets
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${CYAN}💡 Dica: Datasets com ✅ são baixados automaticamente!${NC}"
    echo ""
    read -p "Pressione Enter para voltar ao menu..."
}

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO: LISTAR MODELOS
# ═══════════════════════════════════════════════════════════════

list_models() {
    show_banner
    echo -e "${BOLD}${PURPLE}🤖 MODELOS GAN DISPONÍVEIS${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    python train.py --list-models
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${CYAN}💡 DCGAN: Melhor para iniciantes (rápido e estável)${NC}"
    echo -e "${CYAN}💡 WGAN-GP: Melhor qualidade (mais lento)${NC}"
    echo ""
    read -p "Pressione Enter para voltar ao menu..."
}

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO: AJUDA E DOCUMENTAÇÃO
# ═══════════════════════════════════════════════════════════════

show_help() {
    show_banner
    echo -e "${BOLD}${PURPLE}📖 AJUDA E DOCUMENTAÇÃO${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${BOLD}${GREEN}🚀 INÍCIO RÁPIDO:${NC}"
    echo ""
    echo -e "  ${CYAN}Para primeiro teste (5 minutos):${NC}"
    echo -e "    1. Escolha opção ${GREEN}2${NC} (Exemplos rápidos)"
    echo -e "    2. Escolha opção ${GREEN}1${NC} (Teste super rápido)"
    echo -e "    3. Aguarde ~5 minutos"
    echo -e "    4. Use opção ${GREEN}3${NC} para gerar imagens!"
    echo ""
    
    echo -e "${BOLD}${CYAN}💡 FLUXO COMPLETO:${NC}"
    echo ""
    echo -e "  ${YELLOW}Passo 1:${NC} Treinar modelo"
    echo -e "    → Opção 1 (assistente) ou Opção 2 (exemplos)"
    echo ""
    echo -e "  ${YELLOW}Passo 2:${NC} Aguardar treinamento"
    echo -e "    → Veja progresso no terminal"
    echo ""
    echo -e "  ${YELLOW}Passo 3:${NC} Gerar imagens"
    echo -e "    → Opção 3 (modo automático recomendado)"
    echo ""
    echo -e "  ${YELLOW}Passo 4:${NC} Ver resultados"
    echo -e "    → Abra as imagens geradas em outputs/"
    echo ""
    
    echo -e "${BOLD}${YELLOW}⚙️ COMANDOS DIRETOS (Linha de Comando):${NC}"
    echo ""
    echo -e "  ${CYAN}# Treinar:${NC}"
    echo -e "  python train.py --dataset mnist --model dcgan --epochs 25"
    echo ""
    echo -e "  ${CYAN}# Gerar imagens:${NC}"
    echo -e "  python quick_generate.py"
    echo ""
    echo -e "  ${CYAN}# Ver ajuda completa:${NC}"
    echo -e "  python train.py --help"
    echo ""
    
    echo -e "${BOLD}${PURPLE}🔧 TROUBLESHOOTING:${NC}"
    echo ""
    echo -e "  ${RED}Problema:${NC} CUDA out of memory"
    echo -e "  ${GREEN}Solução:${NC} Reduza batch-size (use 32 ou 64)"
    echo ""
    echo -e "  ${RED}Problema:${NC} Treinamento muito lento"
    echo -e "  ${GREEN}Solução:${NC} Use GPU ou reduza épocas para teste"
    echo ""
    echo -e "  ${RED}Problema:${NC} Imagens ruins"
    echo -e "  ${GREEN}Solução:${NC} Treine por mais épocas ou use WGAN-GP"
    echo ""
    
    echo -e "${BOLD}${BLUE}📚 DOCUMENTAÇÃO COMPLETA:${NC}"
    echo ""
    echo -e "  • README.md - Guia completo do projeto"
    echo -e "  • cat README.md | less"
    echo ""
    
    echo -e "${BOLD}${GREEN}🎯 RECOMENDAÇÕES:${NC}"
    echo ""
    echo -e "  ${YELLOW}Para testes:${NC} MNIST + DCGAN + 5-10 épocas"
    echo -e "  ${YELLOW}Para aprender:${NC} MNIST + DCGAN + 25 épocas"
    echo -e "  ${YELLOW}Para qualidade:${NC} CIFAR-10 + WGAN-GP + 100 épocas"
    echo ""
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    read -p "Pressione Enter para voltar ao menu..."
}

# ═══════════════════════════════════════════════════════════════
# VERIFICAÇÕES INICIAIS
# ═══════════════════════════════════════════════════════════════

check_dependencies() {
    # Verificar Python
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ ERRO: Python não encontrado!${NC}"
        echo "Por favor, instale Python 3.8+ antes de continuar."
        exit 1
    fi
    
    # Verificar PyTorch
    if ! python -c "import torch" &> /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Dependências não encontradas.${NC}"
        echo ""
        read -p "Deseja instalar as dependências agora? [s/N]: " install_deps
        
        if [[ $install_deps =~ ^[Ss]$ ]]; then
            echo ""
            echo -e "${CYAN}📦 Instalando dependências...${NC}"
            pip install -r requirements.txt
            echo ""
            echo -e "${GREEN}✅ Dependências instaladas com sucesso!${NC}"
            sleep 2
        else
            echo ""
            echo -e "${RED}Por favor, instale as dependências primeiro:${NC}"
            echo "  pip install -r requirements.txt"
            echo ""
            exit 1
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════════════

main() {
    while true; do
        show_banner
        show_main_menu
        
        read -p "$(echo -e ${YELLOW}Escolha uma opção: ${NC})" choice
        
        case $choice in
            1) train_model ;;
            2) quick_examples ;;
            3) generate_images ;;
            4) generate_by_class ;;
            5) upscale_images ;;
            6) show_status ;;
            7) list_datasets ;;
            8) list_models ;;
            9) show_help ;;
            0) 
                echo ""
                echo -e "${GREEN}${BOLD}Até logo! 👋${NC}"
                echo ""
                exit 0
                ;;
            *)
                echo -e "${RED}Opção inválida! Tente novamente.${NC}"
                sleep 1
                ;;
        esac
    done
}

# ═══════════════════════════════════════════════════════════════
# EXECUTAR
# ═══════════════════════════════════════════════════════════════

check_dependencies
main
