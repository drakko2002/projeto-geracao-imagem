# IA Image Generator (Offline)

Projeto de faculdade para geração de imagens offline via IA (Stable Diffusion).

## Instalação
```bash
conda create -n sd python=3.10
conda activate sd
pip install -r requirements.txt

docker run -p 5000:5000 \
--env-file local.env \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ia-projeto
