import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class ImageSimilaritySearch:
    """Sistema de busca por similaridade de imagens usando CLIP e scikit-learn"""
    
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.embeddings = None
        self.image_paths = None
        self.image_ids = None
        
    def load_model(self):
        """Carrega o modelo CLIP"""
        print(f"Carregando modelo CLIP no dispositivo: {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("✅ Modelo CLIP carregado")
        return self
    
    def get_image_embedding(self, image_path):
        """Gera embedding para uma imagem"""
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding / embedding.norm()
                embedding = embedding.cpu().numpy().astype('float32')
            
            return embedding
        except Exception as e:
            print(f"Erro ao processar {image_path}: {e}")
            return None
    
    def load_embeddings_from_db(self, db_session, ImageModel):
        """Carrega embeddings do banco de dados"""
        print("Carregando embeddings do banco...")
        
        images = db_session.query(ImageModel).all()
        
        self.embeddings = []
        self.image_paths = []
        self.image_ids = []
        
        for img in images:
            if img.embedding is not None:
                self.embeddings.append(img.embedding)
                self.image_paths.append(img.filepath)
                self.image_ids.append(img.id)
        
        if self.embeddings:
            self.embeddings = np.array(self.embeddings)
            print(f"✅ {len(self.embeddings)} embeddings carregados")
        else:
            print("⚠️  Nenhum embedding encontrado no banco")
        
        return self
    
    def find_similar_images(self, query_embedding, top_k=5):
        """Encontra imagens similares usando cosine similarity"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # Calcular similaridades
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Pegar os top_k mais similares
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'rank': i + 1,
                'similarity': float(similarities[idx]),
                'distance': 1 - float(similarities[idx]),  # Para compatibilidade
                'image_id': self.image_ids[idx],
                'filepath': self.image_paths[idx],
                'filename': os.path.basename(self.image_paths[idx])
            })
        
        return results
    
    def search_by_image(self, image_path, top_k=5):
        """Busca completa: processa imagem e encontra similares"""
        query_embedding = self.get_image_embedding(image_path)
        
        if query_embedding is None:
            return []
        
        return self.find_similar_images(query_embedding, top_k)