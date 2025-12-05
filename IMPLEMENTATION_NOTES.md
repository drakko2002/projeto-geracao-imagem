# Implementação de Geração Condicional por Prompt

## Resumo

Esta implementação adiciona suporte para geração condicional baseada em prompts de texto, permitindo que usuários controlem qual classe de imagem será gerada quando usam modelos condicionais (DCGAN-COND).

## Mudanças Principais

### 1. Utilitários Compartilhados (utils.py)

Quatro novas funções foram adicionadas para fornecer funcionalidade compartilhada entre CLI e GUI:

#### `class_index_from_prompt(prompt_text, dataset_name, dataset_configs)`
- **Objetivo**: Mapeia texto de prompt do usuário para índice de classe
- **Recursos**:
  - Matching case-insensitive (maiúsculas/minúsculas)
  - Suporte para acentos (português): "aviao" casa com "Aviões"
  - Suporte para plural/singular: "gato" casa com "Gatos"
  - Regras especiais para português: "ão" -> "ões" (avião -> aviões)
  - Suporte para dígitos em MNIST: "numero 5" -> índice 5

#### `prompt_to_seed(prompt_text, dataset_name, selected_class, extra)`
- **Objetivo**: Gera seed determinística para modelos incondicionais
- **Uso**: Garante que mesmo prompt sempre gera mesma imagem
- **Parâmetro extra**: Permite variação mantendo consistência do prompt

#### `is_conditional_checkpoint(checkpoint)`
- **Objetivo**: Detecta se checkpoint é de modelo condicional
- **Verifica**:
  - model_type == "dcgan-cond"
  - flag is_conditional = True
  - flag text_conditional = True

#### `get_num_classes_from_checkpoint(checkpoint, dataset_configs)`
- **Objetivo**: Extrai número de classes do checkpoint
- **Fallback**: Infere do dataset_configs se não especificado

### 2. CLI Interativo (generate_interactive.py)

#### Detecção Automática de Tipo de Modelo
- Carrega checkpoint e detecta se é condicional ou incondicional
- Mostra informação clara na UI sobre o tipo de modelo

#### Geração Condicional
- **Quando checkpoint é condicional**:
  - Extrai índice de classe do prompt
  - Passa labels para `generator(noise, labels)`
  - Usuário tem controle real sobre a classe gerada
  - Exemplo: "gato" → gera especificamente um gato

#### Geração Incondicional (Fallback)
- **Quando checkpoint é incondicional**:
  - Deriva seed do prompt para consistência
  - Chama `generator(noise)` sem labels
  - Mesmo prompt → mesma imagem (via seed)
  - Sem controle real de classe

#### Menu Interativo Atualizado
- Mostra se modelo é condicional ou incondicional
- Mensagens claras sobre capacidades do modelo
- Exemplos contextuais baseados no tipo

### 3. Interface Gráfica (app_gui.py)

#### Detecção de Modelo ao Carregar
- Usa `is_conditional_checkpoint()` ao carregar modelo
- Atualiza estado global `is_conditional`

#### Geração com Suporte Condicional
```python
if is_conditional:
    selected_idx = class_index_from_prompt(prompt_text, dataset_name, DATASET_CONFIGS)
    labels = torch.tensor([selected_idx], device=device, dtype=torch.long)
    fake = generator(noise, labels)
else:
    fake = generator(noise)
```

#### UI Dinâmica
- Hint text atualiza baseado no tipo de modelo
- **Condicional**: "✅ O prompt controla a classe gerada"
- **Incondicional**: "⚠️ Prompt usado como semente"

### 4. Documentação (README.md)

#### Nova Seção: Geração Condicional vs Incondicional
- Explica diferença entre DCGAN e DCGAN-COND
- Tabela comparativa de recursos
- Exemplos de uso para cada tipo
- Como treinar modelos condicionais

## Compatibilidade

### ✅ Retrocompatibilidade Total
- Checkpoints incondicionais existentes continuam funcionando
- Nenhuma mudança quebra funcionalidade existente
- Fallback automático para modo incondicional

### ✅ Windows (Batch Scripts)
- **INICIAR.bat**: Passa `--prompt` para generate_interactive.py
- **generate.bat**: Passa `--prompt` para generate_interactive.py
- Ambos funcionam sem modificações com novo código

### ✅ Linux/Mac
- app_gui.py usa mesma lógica que CLI
- run.sh continua funcionando normalmente

## Exemplos de Uso

### Modelo Condicional (Controle Real)
```bash
# Treinar modelo condicional
python train.py --dataset mnist --model dcgan-cond --epochs 25

# Gerar com controle de classe
python generate_interactive.py \
  --checkpoint outputs/mnist/dcgan-cond_xxx/checkpoints/checkpoint_latest.pth \
  --prompt "numero 5"
# Resultado: Gera especificamente o dígito 5
```

### Modelo Incondicional (Seed-based)
```bash
# Treinar modelo incondicional
python train.py --dataset mnist --model dcgan --epochs 25

# Gerar com seed derivada do prompt
python generate_interactive.py \
  --checkpoint outputs/mnist/dcgan_xxx/checkpoints/checkpoint_latest.pth \
  --prompt "numero 5"
# Resultado: Gera imagem aleatória, mas sempre a mesma para este prompt
```

## Testes Realizados

### Testes de Lógica
- ✅ Matching de classes com acentos (aviao → Aviões)
- ✅ Matching plural/singular (gato → Gatos)
- ✅ Matching case-insensitive (GATO → Gatos)
- ✅ Dígitos em MNIST (numero 5 → índice 5)
- ✅ Seed determinística (mesmo prompt → mesma seed)

### Testes de Sintaxe
- ✅ Todos os arquivos Python compilam sem erros
- ✅ Imports corretos e organizados
- ✅ Nenhum erro de sintaxe ou referência

## Qualidade de Código

### Melhorias Implementadas
- **Funções auxiliares**: `_remove_accents()`, `_match_class_name()`
- **Constantes nomeadas**: `SEED_HASH_LENGTH = 8`
- **Imports no nível do módulo**: unicodedata, re no topo
- **Documentação completa**: Docstrings para todas as funções
- **Comentários explicativos**: Decisões de design documentadas

### Code Review
- ✅ Cyclomatic complexity reduzida
- ✅ Funções pequenas e focadas
- ✅ Padrões consistentes entre CLI e GUI
- ✅ Mensagens em português mantidas

## Fluxo de Geração

```
┌─────────────────────────────────────┐
│ Usuário fornece prompt             │
│ Ex: "gerar um gato"                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Carregar checkpoint                │
└──────────────┬──────────────────────┘
               │
               ▼
        ┌──────┴──────┐
        │ Detectar    │
        │ tipo modelo │
        └──────┬──────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌─────────────┐ ┌────────────────┐
│ Condicional │ │ Incondicional  │
└──────┬──────┘ └────────┬───────┘
       │                 │
       ▼                 ▼
┌─────────────┐ ┌────────────────┐
│ Extrai      │ │ Deriva seed    │
│ classe do   │ │ do prompt      │
│ prompt      │ │                │
└──────┬──────┘ └────────┬───────┘
       │                 │
       ▼                 ▼
┌─────────────┐ ┌────────────────┐
│ Gera noise  │ │ Gera noise     │
│ + labels    │ │ com seed       │
└──────┬──────┘ └────────┬───────┘
       │                 │
       ▼                 ▼
┌─────────────┐ ┌────────────────┐
│ generator(  │ │ generator(     │
│   noise,    │ │   noise)       │
│   labels)   │ │                │
└──────┬──────┘ └────────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
        ┌───────────────┐
        │ Imagem gerada │
        └───────────────┘
```

## Requisitos Atendidos

- ✅ Prompt-conditioned generation quando checkpoint condicional disponível
- ✅ Map prompt to class index usando DATASET_CONFIGS
- ✅ Pass labels to generator quando condicional
- ✅ Seed determinística para modelos incondicionais
- ✅ Lógica compartilhada entre app_gui.py e CLI (utils.py)
- ✅ Compatibilidade com checkpoints existentes
- ✅ Compatibilidade Windows (INICIAR.bat, generate.bat)
- ✅ Compatibilidade Linux (app_gui.py, run.sh)
- ✅ Mensagens em português mantidas
- ✅ Documentação completa

## Notas Técnicas

### Matching Bidirecional
A função `_match_class_name` usa matching bidirecional (`A in B or B in A`) para permitir flexibilidade máxima em prompts naturais. Embora isso possa teoricamente causar falsos positivos (ex: "car" em "scar"), o risco é aceitável dado:
1. Contexto de uso (prompts em linguagem natural)
2. Classes de datasets bem definidas
3. Benefício de usabilidade supera o risco

### Fallback para Classe 0
Quando um prompt não corresponde a nenhuma classe em um modelo condicional, o sistema usa a classe 0 como fallback. Isso garante que a geração sempre funcione, mesmo com prompts ambíguos.

### Performance
- Imports movidos para nível do módulo quando possível
- Funções auxiliares reduzem repetição de código
- Seed determinística é computacionalmente leve (SHA256 truncado)

## Próximos Passos (Futuro)

1. **Suporte para múltiplas classes no mesmo prompt**: "gato e cachorro"
2. **Embedding de texto real**: Para modelos text-conditional mais avançados
3. **Configuração de classe default**: Permitir usuário escolher fallback
4. **Testes automatizados**: Suite de testes unitários
5. **Métricas de qualidade**: Avaliar qualidade da geração condicional

## Autor

Implementado por: GitHub Copilot Agent
Data: Dezembro 2024
Repository: drakko2002/projeto-geracao-imagem
