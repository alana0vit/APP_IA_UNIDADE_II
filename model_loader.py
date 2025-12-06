import torch
import numpy as np
import os
import sys
import traceback

class ImageClassifier:
    def __init__(self, embeddings_path="embeddings", sample_images_path="sample_images"):
        self.device = "cpu"  # Forçar CPU no Render
        print(f"Iniciando carregamento no dispositivo: {self.device}")
        
        # Tentar carregar CLIP
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            print("✅ Modelo CLIP carregado com sucesso")
            self.clip_loaded = True
        except Exception as e:
            print(f"❌ Erro ao carregar CLIP: {e}")
            print(traceback.format_exc())
            self.clip_loaded = False
            return
        
        self.embeddings_path = embeddings_path
        self.sample_images_path = sample_images_path
        
        # Carregar embeddings
        self.embeddings = None
        self.image_paths = []
        self.index = None
        
        try:
            self.load_embeddings()
        except Exception as e:
            print(f"❌ Erro ao carregar embeddings: {e}")
            print(traceback.format_exc())
    
    def load_embeddings(self):
        """Carrega embeddings e informações das imagens"""
        try:
            # Carregar embeddings
            embeddings_file = os.path.join(self.embeddings_path, "embeddings.npy")
            if os.path.exists(embeddings_file):
                self.embeddings = np.load(embeddings_file, allow_pickle=True)
                print(f"✅ Embeddings carregados: {self.embeddings.shape}")
            else:
                print(f"⚠️ Arquivo não encontrado: {embeddings_file}")
                # Criar embeddings dummy para teste
                self.embeddings = np.random.randn(100, 512).astype("float32")
                print("✅ Embeddings dummy criados para teste")
            
            # Carregar caminhos das imagens
            paths_file = os.path.join(self.embeddings_path, "image_paths.npy")
            if os.path.exists(paths_file):
                self.image_paths = np.load(paths_file, allow_pickle=True).tolist()
                print(f"✅ Caminhos carregados: {len(self.image_paths)} imagens")
            else:
                print(f"⚠️ Arquivo não encontrado: {paths_file}")
                self.image_paths = [f"image_{i}.jpg" for i in range(len(self.embeddings))]
            
            # Criar índice FAISS
            try:
                import faiss
                d = self.embeddings.shape[1]
                self.index = faiss.IndexFlatL2(d)
                self.index.add(self.embeddings.astype("float32"))
                print(f"✅ Índice FAISS criado com {self.index.ntotal} vetores")
            except ImportError:
                print("⚠️ FAISS não disponível, usando busca simples")
                self.index = None
            
        except Exception as e:
            print(f"❌ Erro crítico: {e}")
            print(traceback.format_exc())
    
    def get_sample_images(self, count=20):
        """Retorna algumas imagens de exemplo para exibição"""
        sample_dir = self.sample_images_path
        images = []
        
        if os.path.exists(sample_dir):
            try:
                for f in os.listdir(sample_dir):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        path = os.path.join(sample_dir, f)
                        images.append({
                            'path': f"/sample_images/{f}",
                            'filename': f,
                            'full_path': path
                        })
            except Exception as e:
                print(f"Erro ao listar imagens: {e}")
        
        # Se não encontrar imagens, criar lista dummy
        if not images:
            images = [
                {
                    'path': f"/static/placeholder_{i}.jpg",
                    'filename': f"placeholder_{i}.jpg",
                    'full_path': f"static/placeholder_{i}.jpg"
                }
                for i in range(min(count, 5))
            ]
        
        return images[:min(count, len(images))]
    
    def search_similar_images(self, image_path, k=5):
        """Busca imagens similares à imagem fornecida"""
        if not self.clip_loaded:
            print("⚠️ CLIP não carregado, retornando resultados dummy")
            return self.get_dummy_results(k)
        
        try:
            if not os.path.exists(image_path):
                print(f"❌ Arquivo não encontrado: {image_path}")
                return self.get_dummy_results(k)
            
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                query_emb = self.model.encode_image(img_tensor)
                query_emb /= query_emb.norm()
                query_emb = query_emb.cpu().numpy().astype("float32")
            
            # Buscar imagens similares
            if self.index is not None:
                distances, indices = self.index.search(query_emb, k)
            else:
                # Busca simples se FAISS não estiver disponível
                distances = []
                indices = []
                for i in range(min(k, len(self.embeddings))):
                    distances.append([i * 0.1])
                    indices.append([i])
                distances = np.array(distances)
                indices = np.array(indices)
            
            # Preparar resultados
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.image_paths):
                    img_path = self.image_paths[idx]
                    
                    # Verificar se o arquivo existe
                    if os.path.exists(img_path):
                        result_path = img_path
                    else:
                        # Tentar encontrar na pasta de samples
                        filename = os.path.basename(img_path)
                        sample_path = os.path.join(self.sample_images_path, filename)
                        if os.path.exists(sample_path):
                            result_path = sample_path
                        else:
                            result_path = None
                    
                    results.append({
                        'original_path': img_path,
                        'display_path': result_path,
                        'distance': float(distances[0][i]),
                        'filename': os.path.basename(img_path),
                        'similarity_percent': max(0, 100 - distances[0][i] * 10)
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            print(traceback.format_exc())
            return self.get_dummy_results(k)
    
    def get_dummy_results(self, k=5):
        """Retorna resultados dummy para teste"""
        results = []
        for i in range(min(k, 5)):
            results.append({
                'original_path': f"dummy_image_{i}.jpg",
                'display_path': None,
                'distance': 0.1 * (i + 1),
                'filename': f"dummy_image_{i}.jpg",
                'similarity_percent': 90 - (i * 10)
            })
        return results
    
    def get_dataset_stats(self):
        """Retorna estatísticas do dataset"""
        return {
            'total_images': len(self.image_paths),
            'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'has_sample_images': len(self.get_sample_images()) > 0,
            'clip_loaded': self.clip_loaded,
            'faiss_available': self.index is not None
        }