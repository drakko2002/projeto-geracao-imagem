"""
Script de teste simplificado para treinar DCGAN
"""

import subprocess
import sys

print("=" * 60)
print("TESTE DE TREINAMENTO DCGAN")
print("=" * 60)
print("\nIniciando treinamento com:")
print("- Dataset: fake (dados sintéticos)")
print("- Épocas: 2")
print("- Modo: dry-run (apenas 1 batch por época)")
print("- Output: outputs/")
print("\n" + "=" * 60 + "\n")

cmd = [
    sys.executable,
    "dcgan/main.py",
    "--dataset",
    "fake",
    "--niter",
    "2",
    "--dry-run",
    "--outf",
    "outputs/",
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print(f"\nCódigo de saída: {result.returncode}")

    if result.returncode == 0:
        print("\n✓ Treinamento concluído com sucesso!")
        print("\nVerifique a pasta 'outputs/' para ver as imagens geradas.")
    else:
        print("\n✗ Erro durante o treinamento")

except subprocess.TimeoutExpired:
    print("\n⚠ Timeout - treinamento demorou mais de 2 minutos")
except Exception as e:
    print(f"\n✗ Erro: {e}")
