# check_sizes.py
import os
import numpy as np

def check_file_sizes():
    total_size = 0
    
    # Verificar embeddings
    if os.path.exists("embeddings/embeddings.npy"):
        emb_size = os.path.getsize("embeddings/embeddings.npy") / 1024 / 1024
        print(f"embeddings.npy: {emb_size:.2f} MB")
        total_size += emb_size
    
    # Verificar sample images
    sample_size = 0
    if os.path.exists("sample_images"):
        for root, dirs, files in os.walk("sample_images"):
            for f in files:
                path = os.path.join(root, f)
                sample_size += os.path.getsize(path)
        print(f"sample_images/: {sample_size/1024/1024:.2f} MB")
        total_size += sample_size / 1024 / 1024
    
    print(f"\n📊 Tamanho total aproximado: {total_size:.2f} MB")
    
    # Verificar se cabe no Render Free (512MB)
    if total_size > 400:  # Deixar margem para o sistema
        print("⚠️  AVISO: Pode não caber no plano Free do Render!")
        print("   Considere reduzir o número de embeddings ou comprimir as imagens.")
    else:
        print("✅ Tamanho adequado para Render Free")
    
    return total_size

if __name__ == "__main__":
    check_file_sizes()