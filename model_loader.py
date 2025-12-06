import torch
import clip
import numpy as np
import faiss
from PIL import Image
import os
import pickle
import pandas as pd
from tqdm import tqdm
import sys

try:
    import clip
    CLIP_AVAILABLE = True
    print("✅ CLIP importado com sucesso")
except ImportError as e:
    print(f"⚠️  CLIP não pôde ser importado: {e}")
    print("⚠️  Instalando CLIP...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
    import clip
    CLIP_AVAILABLE = True
    print("✅ CLIP instalado e importado")

try:
    import faiss
    FAISS_AVAILABLE = True
    print("✅ FAISS importado com sucesso")
except ImportError as e:
    print(f"❌ FAISS não pôde ser importado: {e}")
    FAISS_AVAILABLE = False

class ImageClassifier:
    def __init__(self, embeddings_path="embeddings", sample_images_path="sample_images"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {self.device}")
        
        # Carregar modelo CLIP
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.embeddings_path = embeddings_path
        self.sample_images_path = sample_images_path
        
        # Carregar embeddings
        self.embeddings = None
        self.image_paths = []
        self.image_info = {}
        self.index = None
        
        self.load_embeddings()
    
    def load_embeddings(self):
        """Carrega embeddings e informações das imagens"""
        try:
            # Carregar embeddings
            embeddings_file = os.path.join(self.embeddings_path, "embeddings.npy")
            if os.path.exists(embeddings_file):
                self.embeddings = np.load(embeddings_file, allow_pickle=True)
                print(f"✅ Embeddings carregados: {self.embeddings.shape}")
            else:
                print("❌ Arquivo de embeddings não encontrado!")
                return
            
            # Carregar caminhos das imagens
            paths_file = os.path.join(self.embeddings_path, "image_paths.npy")
            if os.path.exists(paths_file):
                self.image_paths = np.load(paths_file, allow_pickle=True).tolist()
                print(f"✅ Caminhos carregados: {len(self.image_paths)} imagens")
            
            # Carregar informações das imagens
            info_file = os.path.join(self.embeddings_path, "image_info.csv")
            if os.path.exists(info_file):
                df_info = pd.read_csv(info_file)
                for _, row in df_info.iterrows():
                    self.image_info[row['path']] = {
                        'filename': row['filename'],
                        'folder': row['folder'],
                        'size': row['size']
                    }
            
            # Criar índice FAISS
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.embeddings.astype("float32"))
            
            print(f"✅ Índice FAISS criado com {self.index.ntotal} vetores")
            
        except Exception as e:
            print(f"❌ Erro ao carregar embeddings: {e}")
    
    def get_sample_images(self, count=20):
        """Retorna algumas imagens de exemplo para exibição"""
        sample_dir = self.sample_images_path
        if os.path.exists(sample_dir):
            images = []
            for f in os.listdir(sample_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(sample_dir, f)
                    rel_path = os.path.relpath(path, start='.')
                    images.append({
                        'path': rel_path,
                        'filename': f,
                        'full_path': path
                    })
            
            # Retornar no máximo 'count' imagens
            return images[:min(count, len(images))]
        return []
    
    def search_similar_images(self, image_path, k=5):
        """Busca imagens similares à imagem fornecida"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ Arquivo não encontrado: {image_path}")
                return []
            
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                query_emb = self.model.encode_image(img_tensor)
                query_emb /= query_emb.norm()
                query_emb = query_emb.cpu().numpy().astype("float32")
            
            # Buscar imagens similares
            distances, indices = self.index.search(query_emb, k)
            
            # Preparar resultados
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.image_paths):
                    img_path = self.image_paths[idx]
                    
                    # Verificar se o arquivo original existe
                    if os.path.exists(img_path):
                        result_path = img_path
                    else:
                        # Se não existir, usar o nome do arquivo para buscar na pasta de samples
                        filename = os.path.basename(img_path)
                        sample_path = os.path.join(self.sample_images_path, filename)
                        if os.path.exists(sample_path):
                            result_path = sample_path
                        else:
                            continue  # Pular se não encontrar
                    
                    results.append({
                        'original_path': img_path,
                        'display_path': result_path,
                        'distance': float(distances[0][i]),
                        'filename': os.path.basename(img_path),
                        'similarity_percent': max(0, 100 - distances[0][i] * 10)  # Converter para porcentagem
                    })
            
            return results
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []
    
    def get_dataset_stats(self):
        """Retorna estatísticas do dataset"""
        return {
            'total_images': len(self.image_paths),
            'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'has_sample_images': len(self.get_sample_images()) > 0
        }