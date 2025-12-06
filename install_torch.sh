#!/bin/bash
# install_torch.sh

echo "Instalando PyTorch para CPU..."
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

echo "Verificando instalação do PyTorch..."
python -c "import torch; print(f'PyTorch versão: {torch.__version__}'); print(f'Dispositivo disponível: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"