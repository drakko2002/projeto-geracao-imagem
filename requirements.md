#Para placa 940mx NVIDIA
pip install diffusers==0.10.2
pip install transformers==4.25.1
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

#Pro caso de preferir a versao conda.
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -c nvidia

#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.#12.1 cudatoolkit=11.3 -c pytorch -c nvidia
