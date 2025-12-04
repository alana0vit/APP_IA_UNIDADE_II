import os
import sys
import torch
import clip
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Configurações
app = Flask(__name__)
app.secret_key = 'sua-chave-secreta-aqui'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Variáveis globais
model = None
preprocess = None
embeddings_cache = None
image_paths_cache = []

def load_model():
    """Carrega o modelo CLIP"""
    global model, preprocess
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Carregando modelo CLIP no dispositivo: {device}")
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✅ Modelo CLIP carregado")
        return True
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

def download_cache_from_url():
    """Baixa o cache de uma URL externa"""
    import urllib.request
    import gzip
    
    cache_url = "https://drive.google.com/file/d/19JLPivjrLH3kthbzEJeCVcpZH0p5jhr8/view?usp=sharing"
    
    print(f"Baixando cache de {cache_url}...")
    
    try:
        # Baixar arquivo
        urllib.request.urlretrieve(cache_url, "embeddings_cache.pkl.gz")
        
        # Descomprimir
        with gzip.open("embeddings_cache.pkl.gz", 'rb') as f:
            import pickle
            data = pickle.load(f)
        
        # Salvar local
        with open("embeddings_cache.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print("✅ Cache baixado e extraído")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao baixar cache: {e}")
        return False

def load_embeddings():
    """Carrega embeddings - com fallback para download"""
    global embeddings_cache, image_paths_cache
    
    cache_file = "embeddings_cache.pkl"
    
    if os.path.exists(cache_file):
        print("Carregando embeddings do cache local...")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            embeddings_cache = data['embeddings']
            image_paths_cache = data['image_paths']
        print(f"✅ {len(image_paths_cache)} embeddings carregados")
        return True
    else:
        print("⚠️  Cache local não encontrado")
        
        # Tentar baixar
        if download_cache_from_url():
            return load_embeddings()  # Recarregar
        else:
            print("❌ Não foi possível obter o cache")
            return False

def find_similar_images(query_image_path, top_k=5):
    """Encontra imagens similares"""
    try:
        # Processar imagem de consulta
        query_image = Image.open(query_image_path).convert("RGB")
        query_tensor = preprocess(query_image).unsqueeze(0).to("cpu")
        
        with torch.no_grad():
            query_embedding = model.encode_image(query_tensor)
            query_embedding = query_embedding / query_embedding.norm()
            query_embedding = query_embedding.cpu().numpy()
        
        # Calcular similaridades
        similarities = cosine_similarity(query_embedding, embeddings_cache)[0]
        
        # Pegar top K
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Preparar resultados
        results = []
        for i, idx in enumerate(top_indices):
            # Obter caminho relativo para a web
            full_path = image_paths_cache[idx]
            rel_path = os.path.relpath(full_path, 'epillid')
            
            results.append({
                'rank': i + 1,
                'similarity': float(similarities[idx]),
                'filepath': full_path,  # Caminho completo (para processamento)
                'web_path': f"/dataset_image/{rel_path}",  # Caminho para web
                'filename': os.path.basename(full_path)
            })
        
        return results
        
    except Exception as e:
        print(f"Erro na busca: {e}")
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    """Página principal"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Selecione uma imagem!')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('Selecione uma imagem!')
            return redirect(request.url)
        
        if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Salvar imagem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            
            # Buscar similares
            results = find_similar_images(filepath, top_k=5)
            
            if results:
                return render_template('results.html',
                                     query_image=filename,
                                     results=results)
            else:
                flash('Nenhuma similar encontrada!')
                return redirect(request.url)
        else:
            flash('Formato inválido! Use JPG ou PNG.')
            return redirect(request.url)
    
    return render_template('index.html',
                         total_images=len(image_paths_cache) if image_paths_cache else 0)

@app.route('/dataset_image/<path:filename>')
def dataset_image(filename):
    """Serve imagens do dataset"""
    try:
        # Procurar a imagem no dataset
        dataset_path = "epillid"
        image_path = os.path.join(dataset_path, filename)
        
        if os.path.exists(image_path):
            return send_file(image_path)
        else:
            # Tentar encontrar pelo caminho completo
            for img_path in image_paths_cache:
                if filename in img_path:
                    return send_file(img_path)
            
            return "Imagem não encontrada", 404
    except Exception as e:
        print(f"Erro ao servir imagem {filename}: {e}")
        return "Erro", 500
    
if __name__ == '__main__':
    # Criar pasta de uploads
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("🚀 Iniciando E-Pill Finder...")
    
    if load_model():
        if load_embeddings():
            print(f"✅ Sistema pronto com {len(image_paths_cache)} imagens")
            
            # Configuração para Render
            port = int(os.environ.get('PORT', 5000))
            debug_mode = os.environ.get('FLASK_ENV') == 'development'
            
            app.run(host='0.0.0.0', port=port, debug=debug_mode)
        else:
            print("❌ Falha ao carregar embeddings")
    else:
        print("❌ Falha ao carregar modelo")
