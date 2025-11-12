import os
import glob
import re
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

import torch

from models import get_model
from config import DATASET_CONFIGS

# -------------------------------------------------------
# Descobrir modelos disponíveis (MNIST, CIFAR10, etc)
# -------------------------------------------------------

def find_available_models():
    """
    Vasculha outputs/**/checkpoint_latest.pth e, para cada dataset
    (pasta logo após 'outputs/'), escolhe o checkpoint mais recente.
    Retorna: { dataset_name: ckpt_path_mais_recente }
    """
    found = {}
    for ckpt in glob.glob(os.path.join("outputs", "**", "checkpoint_latest.pth"), recursive=True):
        parts = ckpt.replace("/", os.sep).split(os.sep)
        if len(parts) >= 3 and parts[0] == "outputs":
            ds = parts[1]  # outputs/<dataset>/...
            try:
                mtime = os.path.getmtime(ckpt)
            except OSError:
                continue
            if ds not in found or mtime > found[ds][1]:
                found[ds] = (ckpt, mtime)
    # só o caminho
    return {ds: data[0] for ds, data in found.items()}

AVAILABLE_MODELS = find_available_models()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# estado global simples
current_dataset = None
current_checkpoint = None
generator = None
nz = 100
generation_counter = 0
# controle lógico do tipo de modelo e classes
is_conditional = False   # True para modelos "dcgan-cond"
classes_map = []         # nomes de classe do dataset atual (DATASET_CONFIGS)

# -------------------------------------------------------
# Utilidades de prompt
# -------------------------------------------------------

def parse_prompt(prompt, dataset_name):
    """
    Lê o prompt e tenta mapear para uma 'classe' desse dataset.
    Isso é só para derivar a seed; o modelo continua sendo incondicional.
    """
    if not prompt or dataset_name not in DATASET_CONFIGS:
        return None

    classes = DATASET_CONFIGS[dataset_name].get("classes", [])
    prompt_lower = prompt.lower()

    # match por substring
    for cls in classes:
        if cls.lower() in prompt_lower:
            return cls

    # MNIST: número no texto
    if dataset_name == "mnist":
        nums = re.findall(r"\d", prompt)
        if nums:
            return nums[0]

    return None

def class_index_from_prompt(prompt_text: str, dataset_name: str) -> int | None:
    """
    Retorna índice de classe a partir do texto do usuário.
    - Casa por substring com DATASET_CONFIGS[dataset]['classes'].
    - Para MNIST, aceita o primeiro dígito no texto.
    """
    classes = DATASET_CONFIGS.get(dataset_name, {}).get("classes", [])
    if not classes or not prompt_text:
        return None

    text = prompt_text.lower()

    # 1) por nome de classe (CIFAR-10, Fashion-MNIST etc.)
    for i, cname in enumerate(classes):
        if cname and cname.lower() in text:
            return i

    # 2) MNIST: aceita dígito
    if dataset_name == "mnist":
        nums = re.findall(r"\d", prompt_text)
        if nums:
            d = int(nums[0])
            if 0 <= d < len(classes):
                return d

    return None

def prompt_to_seed(prompt_text, dataset_name, selected_class, extra=0):
    """
    Gera uma seed a partir do prompt + dataset + classe + extra.
    O 'extra' é usado para variar a cada clique,
    mantendo o prompt ainda como parte da chave.
    """
    base = f"{dataset_name}|{selected_class or ''}|{prompt_text}|{extra}"
    import hashlib
    h = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32 bits já são suficientes


# -------------------------------------------------------
# Carregar modelo
# -------------------------------------------------------

def load_generator(dataset_name):
    """
    Carrega o gerador do dataset escolhido.
    """
    global generator, nz, current_dataset, current_checkpoint

    if dataset_name not in AVAILABLE_MODELS:
        messagebox.showerror(
            "Modelo não encontrado",
            f"Nenhum checkpoint encontrado para o dataset '{dataset_name}'.\n"
            f"Use seu instalador/menu para baixar esse modelo primeiro."
        )
        return False

    ckpt_path = AVAILABLE_MODELS[dataset_name]

    if (
        generator is not None
        and current_dataset == dataset_name
        and current_checkpoint == ckpt_path
    ):
        # já carregado
        return True

    if not os.path.exists(ckpt_path):
        messagebox.showerror(
            "Checkpoint ausente",
            f"O caminho do checkpoint não existe:\n{ckpt_path}"
        )
        return False

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        messagebox.showerror("Erro ao carregar modelo", str(e))
        return False

    config = ckpt.get("config", {})
    model_type = config.get("model", "dcgan")

    nz_local = config.get("nz", 100)
    ngf = config.get("ngf", 64)
    ndf = config.get("ndf", 64)
    ds_name = config.get("dataset", dataset_name)

    # sinaliza se o checkpoint é condicional e carrega as classes do dataset
    global is_conditional, classes_map
    is_conditional = (str(model_type).lower() == "dcgan-cond") or bool(config.get("is_conditional", False))
    classes_map = DATASET_CONFIGS.get(ds_name, {}).get("classes", [])

    # canais / tamanho baseado no checkpoint
    nc = config.get("nc", 1 if ds_name == "mnist" else 3)
    img_size = config.get("img_size", 28 if ds_name == "mnist" else 32)

    # ---------- TUDO ABAIXO PRECISA ESTAR DENTRO DA FUNÇÃO ----------
    # Se for condicional, precisamos de num_classes (e opcionalmente text_conditional)
    num_classes = None
    text_conditional = False

    # monta a config base do gerador
    model_config = {
        "nz": nz_local,
        "ngf": ngf,
        "ndf": ndf,
        "nc": nc,
        "img_size": img_size,
    }

    # se o checkpoint/modelo for condicional, garantimos num_classes
    if is_conditional:
        num_classes = config.get("num_classes", None)
        text_conditional = bool(config.get("text_conditional", False))

        # tenta inferir pelo mapeamento de classes do dataset (ex.: 10 no MNIST/CIFAR10)
        if num_classes is None and classes_map:
            num_classes = len(classes_map)

        if num_classes is None:
            messagebox.showerror(
                "Erro ao inicializar gerador",
                "num_classes é obrigatório para dcgan-cond"
            )
            return False

        model_config.update({
            "num_classes": num_classes,
            "text_conditional": text_conditional,
        })

    # cria o gerador (condicional ou não) com a config montada
    try:
        gen, _ = get_model(model_type, model_config)
        gen.load_state_dict(ckpt["generator_state_dict"])
        gen.to(device)
        gen.eval()
    except Exception as e:
        messagebox.showerror("Erro ao inicializar gerador", str(e))
        return False

    # atualiza estado global só no final (se tudo deu certo)
    nz = nz_local
    generator = gen
    current_dataset = ds_name
    current_checkpoint = ckpt_path
    return True


# -------------------------------------------------------
# Geração de imagem
# -------------------------------------------------------

def generate_image(prompt_text, image_label, dataset_var):
    global generator, generation_counter

    dataset_name = dataset_var.get()

    if not dataset_name:
        messagebox.showwarning("Selecione um modelo", "Escolha um dataset antes de gerar.")
        return

    if not load_generator(dataset_name):
        return

    # interpretar prompt dentro do dataset escolhido
    selected_class = parse_prompt(prompt_text, dataset_name)

    # incrementa contador para variar seed a cada geração
    generation_counter += 1
    seed = prompt_to_seed(prompt_text, dataset_name, selected_class, extra=generation_counter)

    try:
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        noise = torch.randn(1, nz, 1, 1, generator=g, device=device)

        with torch.no_grad():
            if is_conditional:
                # extrai índice de classe a partir do prompt; default=0 se não achou
                selected_idx = class_index_from_prompt(prompt_text, dataset_name)
                if selected_idx is None:
                    selected_idx = 0
                labels = torch.tensor([selected_idx], device=device, dtype=torch.long)
                fake = generator(noise, labels).detach().cpu()
            else:
                fake = generator(noise).detach().cpu()

        fake = (fake + 1) / 2  # [-1,1] -> [0,1]
        fake = fake.squeeze(0)

        # grayscale vs RGB
        if fake.shape[0] == 1:
            img_np = fake[0].numpy()
            img = Image.fromarray((img_np * 255).astype("uint8"), mode="L")
        else:
            img_np = fake.permute(1, 2, 0).numpy()
            img = Image.fromarray((img_np * 255).astype("uint8"), mode="RGB")

        # tamanho maior pra ficar mais bonito na UI
        img = img.resize((340, 340), Image.NEAREST)

        tk_img = ImageTk.PhotoImage(img)
        image_label.config(image=tk_img, text="")
        image_label.image = tk_img

    except Exception as e:
        messagebox.showerror("Erro ao gerar imagem", str(e))

# -------------------------------------------------------
# Interface Tkinter (somente estética)
# -------------------------------------------------------

def main():
    if not AVAILABLE_MODELS:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Nenhum modelo encontrado",
            "Nenhum checkpoint_latest.pth foi encontrado em outputs/.\n"
            "Use o instalador ou o script download_models.py para baixar modelos."
        )
        return

    # ---------- janela base ----------
    root = tk.Tk()
    root.title("Gerador de Imagens IA")
    # janela um pouco maior, estilo dashboard
    root.geometry("800x560")
    root.minsize(800, 560)
    root.configure(bg="#0f172a")  # azul escuro

    # ---------- container externo ----------
    outer = tk.Frame(
        root,
        bg="#0f172a",
        padx=30,
        pady=30,
    )
    outer.pack(fill="both", expand=True)

    # ---------- "card" central ----------
    card = tk.Frame(
        outer,
        bg="#111827",           # cinza-azulado escuro
        bd=0,
        highlightthickness=0,
    )
    card.pack(fill="both", expand=True)

    # ---------- header ----------
    header = tk.Frame(card, bg="#111827")
    header.pack(fill="x", padx=24, pady=(20, 10))

    title_label = tk.Label(
        header,
        text="Gerador de Imagens IA",
        bg="#111827",
        fg="#e5e7eb",
        font=("Segoe UI", 18, "bold"),
        anchor="w",
    )
    title_label.pack(side="left")

    subtitle_label = tk.Label(
        header,
        text="Escolha o modelo, escreva um prompt e gere amostras do seu GAN.",
        bg="#111827",
        fg="#9ca3af",
        font=("Segoe UI", 9),
        anchor="w",
    )
    subtitle_label.pack(side="left", padx=(12, 0))

    # ---------- conteúdo principal ----------
    content = tk.Frame(card, bg="#111827")
    content.pack(fill="both", expand=True, padx=24, pady=(10, 20))

    # coluna da imagem
    image_container = tk.Frame(content, bg="#111827")
    image_container.pack(side="left", fill="both", expand=True)

    image_frame = tk.Frame(
        image_container,
        bg="#111827",
        bd=0,
        highlightbackground="#1f2937",
        highlightthickness=1,   # borda sutil
    )
    image_frame.place(relx=0.5, rely=0.5, anchor="center", width=360, height=360)

    image_label = tk.Label(
        image_frame,
        bg="#111827",
        fg="#6b7280",
        text="Pré-visualização da imagem",
        font=("Segoe UI", 10),
    )
    image_label.place(relx=0.5, rely=0.5, anchor="center")

    # coluna de controles
    controls = tk.Frame(content, bg="#111827")
    controls.pack(side="right", fill="y", padx=(24, 0))

    # seleção de modelo/dataset
    lbl_model = tk.Label(
        controls,
        text="Modelo / Dataset",
        bg="#111827",
        fg="#9ca3af",
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    )
    lbl_model.pack(fill="x")

    dataset_var = tk.StringVar()
    dataset_var.set(list(AVAILABLE_MODELS.keys())[0])

    option = tk.OptionMenu(controls, dataset_var, *AVAILABLE_MODELS.keys())
    option.config(
        bg="#111827",
        fg="#e5e7eb",
        activebackground="#1f2937",
        activeforeground="#e5e7eb",
        highlightthickness=0,
        bd=1,
        relief="flat",
        font=("Segoe UI", 9),
    )
    option["menu"].config(
        bg="#111827",
        fg="#e5e7eb",
        activebackground="#1f2937",
        activeforeground="#e5e7eb",
        font=("Segoe UI", 9),
    )
    option.pack(fill="x", pady=(4, 16))

    # prompt label
    lbl_prompt = tk.Label(
        controls,
        text="Prompt (usado como semente)",
        bg="#111827",
        fg="#9ca3af",
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    )
    lbl_prompt.pack(fill="x")

    # contêiner para dar padding interno no Entry (borda "invisível")
    prompt_container = tk.Frame(
        controls,
        bg="#111827",
        bd=0,
        highlightthickness=0,
    )
    prompt_container.pack(fill="x", pady=(4, 16))

    # Entry com espaçamento lateral (texto não fica colado na borda)
    prompt_entry = tk.Entry(
        prompt_container,
        bd=0,
        bg="#111827",
        fg="#e5e7eb",
        insertbackground="#9ca3af",
        font=("Segoe UI", 9),
        highlightthickness=1,
        highlightbackground="#374151",  # borda sutil
        highlightcolor="#2563eb",       # cor ao focar
        relief="flat",
    )
    # padx cria o "padding" entre texto e borda visual
    prompt_entry.pack(fill="x", padx=8, ipady=6)

    # botão gerar
    def on_generate():
        prompt_text = prompt_entry.get().strip() or "imagem aleatoria"
        generate_image(prompt_text, image_label, dataset_var)

    generate_button = tk.Button(
        controls,
        text="Gerar imagem",
        command=on_generate,
        bg="#2563eb",
        fg="#f9fafb",
        activebackground="#1d4ed8",
        activeforeground="#f9fafb",
        bd=0,
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=16,
        pady=8,
        cursor="hand2",
    )
    generate_button.pack(fill="x")

    # dica
    hint_label = tk.Label(
        controls,
        text="Dica: o prompt é usado como semente.\n"
             "Mesmo prompt → imagem consistente\n"
             "Prompts diferentes → variações.",
        bg="#111827",
        fg="#6b7280",
        font=("Segoe UI", 8),
        justify="left",
        anchor="w",
    )
    hint_label.pack(fill="x", pady=(8, 0))

    root.mainloop()


if __name__ == "__main__":
    main()
