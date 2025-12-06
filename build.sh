#!/bin/bash
# build.sh

echo "=== INICIANDO BUILD NO RENDER ==="

# Atualizar pip
python -m pip install --upgrade pip

# Instalar dependências básicas primeiro
echo "1. Instalando dependências básicas..."
pip install Flask==2.3.2
pip install Pillow==9.5.0
pip install numpy==1.24.3
pip install python-dotenv==1.0.0

# Instalar PyTorch CPU
echo "2. Instalando PyTorch CPU..."
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Instalar FAISS
echo "3. Instalando FAISS..."
pip install faiss-cpu==1.7.4

# Instalar CLIP
echo "4. Instalando CLIP..."
pip install git+https://github.com/openai/CLIP.git

# Instalar resto
echo "5. Instalando demais dependências..."
pip install ftfy==6.1.1
pip install regex==2023.6.3
pip install tqdm==4.65.0
pip install pandas==1.5.3
pip install gunicorn==20.1.0
pip install Flask-SQLAlchemy==3.0.5

# Verificar instalação
echo "6. Verificando instalação..."
python -c "
import flask
print(f'Flask: {flask.__version__}')
import PIL
print(f'Pillow: {PIL.__version__}')
import torch
print(f'PyTorch: {torch.__version__}')
import numpy
print(f'NumPy: {numpy.__version__}')
print('✅ Todas as dependências instaladas!')
"

echo "=== BUILD CONCLUÍDO ==="