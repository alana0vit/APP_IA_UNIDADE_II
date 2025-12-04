import torch
import clip
import numpy as np
import faiss
from PIL import Image
import os
from tqdm import tqdm
import pickle

class ImageClassifier:
    def __init__(self, dataset_path="epillid"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {self.device}")
        
        # Carregar modelo CLIP
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.dataset_path = dataset_path
        self.image_paths = []
        self.embeddings = None
        self.index = None
        
        # Verificar se já existem embeddings salvos
        if os.path.exists("embeddings.pkl") and os.path.exists("image_paths.pkl"):
            print("Carregando embeddings salvos...")
            self.load_embeddings()
        else:
            print("Gerando embeddings do dataset...")
            self.prepare_embeddings()
    
    def prepare_embeddings(self):
        """Extrai embeddings do dataset de imagens"""
        # Coletar caminhos das imagens
        for root, dirs, files in os.walk(self.dataset_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(root, f)
                    self.image_paths.append(path)
        
        print(f"Encontradas {len(self.image_paths)} imagens")
        
        # Gerar embeddings
        embeddings_list = []
        for path in tqdm(self.image_paths, desc="Processando imagens"):
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    emb = self.model.encode_image(img_tensor)
                    emb /= emb.norm()
                    embeddings_list.append(emb.cpu().numpy())
            except Exception as e:
                print(f"Erro ao processar {path}: {e}")
                continue
        
        self.embeddings = np.concatenate(embeddings_list, axis=0).astype("float32")
        
        # Criar índice FAISS
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)
        
        # Salvar embeddings para uso futuro
        self.save_embeddings()
    
    def save_embeddings(self):
        """Salva embeddings e caminhos das imagens"""
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)
        
        with open("image_paths.pkl", "wb") as f:
            pickle.dump(self.image_paths, f)
    
    def load_embeddings(self):
        """Carrega embeddings e caminhos das imagens salvos"""
        with open("embeddings.pkl", "rb") as f:
            self.embeddings = pickle.load(f)
        
        with open("image_paths.pkl", "rb") as f:
            self.image_paths = pickle.load(f)
        
        # Recriar índice FAISS
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)
    
    def search_similar_images(self, image_path, k=5):
        """Busca imagens similares à imagem fornecida"""
        try:
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
                    results.append({
                        'path': self.image_paths[idx],
                        'distance': float(distances[0][i]),
                        'filename': os.path.basename(self.image_paths[idx])
                    })
            
            return results
        except Exception as e:
            print(f"Erro na busca: {e}")
            return []
    
    def get_image_info(self, image_path):
        """Obtém informações sobre uma imagem"""
        try:
            img = Image.open(image_path)
            return {
                'size': img.size,
                'mode': img.mode,
                'format': img.format
            }
        except:
            return None