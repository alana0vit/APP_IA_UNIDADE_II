import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss

def generate_embeddings():
    """Gera embeddings para todas as imagens do dataset"""
    print("Gerando embeddings...")
    
    # Configurar dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    
    # Carregar modelo CLIP
    print("Carregando modelo CLIP...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Encontrar todas as imagens
    print("Buscando imagens...")
    image_paths = []
    embeddings = []
    
    dataset_path = "epillid"  # Ajuste conforme seu dataset
    
    if not os.path.exists(dataset_path):
        print(f"Erro: Diretório {dataset_path} não encontrado!")
        print("Certifique-se de que o dataset está na raiz do projeto")
        return False
    
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, f)
                image_paths.append(path)
    
    print(f"Encontradas {len(image_paths)} imagens")
    
    # Gerar embeddings
    for path in tqdm(image_paths, desc="Processando imagens"):
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = model.encode_image(img_tensor)
                emb /= emb.norm()
                embeddings.append(emb.cpu().numpy())
                
        except Exception as e:
            print(f"Erro ao processar {path}: {e}")
            # Adicionar embedding zero para manter o alinhamento
            embeddings.append(np.zeros((1, 512)))  # 512 é a dimensão do ViT-B/32
    
    # Concatenar embeddings
    embeddings_array = np.concatenate(embeddings, axis=0).astype("float32")
    
    # Salvar embeddings e paths
    print("Salvando embeddings...")
    np.save("model/embeddings.npy", embeddings_array)
    np.save("model/image_paths.npy", np.array(image_paths, dtype=object))
    
    print(f"Embeddings salvas: {embeddings_array.shape}")
    print(f"Caminhos das imagens: {len(image_paths)}")
    
    return True

if __name__ == "__main__":
    # Criar diretório se não existir
    os.makedirs("model", exist_ok=True)
    
    # Gerar embeddings
    success = generate_embeddings()
    
    if success:
        print("Embeddings geradas com sucesso!")
        print("Agora você pode executar a aplicação Flask.")
    else:
        print("Falha ao gerar embeddings.")