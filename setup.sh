#!/bin/bash
# setup.sh
set -e

echo "Instalando dependências Python..."
pip install --upgrade pip
pip install wheel setuptools

# Instalar dependências básicas primeiro
pip install Flask==2.3.2
pip install Pillow==9.5.0
pip install numpy==1.24.3

# Instalar PyTorch com CPU-only (mais leve)
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Instalar outras dependências
pip install faiss-cpu==1.7.4
pip install ftfy==6.1.1
pip install regex==2023.6.3
pip install tqdm==4.65.0
pip install pandas==1.5.3
pip install python-dotenv==1.0.0
pip install gunicorn==20.1.0
pip install Flask-SQLAlchemy==3.0.5

# Instalar CLIP
echo "Instalando CLIP..."
pip install git+https://github.com/openai/CLIP.git

echo "Criando estrutura de pastas..."
mkdir -p uploads embeddings sample_images static/css static/js templates

echo "✅ Instalação concluída!"