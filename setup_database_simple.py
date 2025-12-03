import os
import sys
from pathlib import Path

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from model_utils import ImageSimilaritySearch
from database_simple import db, ImageEmbedding, init_db
from tqdm import tqdm

def setup_database():
    """Configura o banco de dados com as imagens do dataset"""
    
    print("=" * 60)
    print("SETUP DO BANCO DE DADOS - E-Pill Classifier")
    print("=" * 60)
    
    # Criar app Flask temporária
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'temp-key-for-setup'
    
    # Inicializar banco
    init_db(app)
    
    # Verificar dataset
    dataset_path = "epillid"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado em: {dataset_path}")
        print(f"📂 Crie uma pasta 'epillid' e coloque suas imagens dentro")
        print(f"   Estrutura sugerida: epillid/classe1/imagem1.jpg")
        return False
    
    # Contar imagens
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    print(f"📊 Encontradas {len(image_files)} imagens")
    
    if len(image_files) == 0:
        print("❌ Nenhuma imagem encontrada")
        return False
    
    # Carregar modelo
    print("\n🚀 Carregando modelo CLIP...")
    search_system = ImageSimilaritySearch()
    search_system.load_model()
    
    with app.app_context():
        # Verificar se já tem dados
        existing_count = db.session.query(ImageEmbedding).count()
        
        if existing_count > 0:
            print(f"⚠️  Já existem {existing_count} imagens no banco")
            response = input("Deseja recriar? (s/n): ").strip().lower()
            if response == 's':
                db.session.query(ImageEmbedding).delete()
                db.session.commit()
                print("Banco limpo")
            else:
                print("Mantendo dados existentes")
                return True
        
        # Processar imagens
        print(f"\n🔧 Processando {len(image_files)} imagens...")
        
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(tqdm(image_files, desc="Processando")):
            try:
                # Extrair nome da classe da estrutura de pastas
                rel_path = os.path.relpath(image_path, dataset_path)
                class_name = os.path.dirname(rel_path)
                if not class_name or class_name == '.':
                    class_name = "unknown"
                
                # Gerar embedding
                embedding = search_system.get_image_embedding(image_path)
                
                if embedding is not None:
                    # Salvar no banco
                    image_record = ImageEmbedding(
                        filename=os.path.basename(image_path),
                        filepath=image_path,
                        class_name=class_name,
                        embedding=embedding.flatten().tolist()  # Salvar como lista
                    )
                    
                    db.session.add(image_record)
                    successful += 1
                    
                    # Commit a cada 50 imagens
                    if successful % 50 == 0:
                        db.session.commit()
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"\n⚠️  Erro em {image_path}: {e}")
                failed += 1
        
        # Commit final
        db.session.commit()
        
        print(f"\n✅ Processamento concluído!")
        print(f"   Sucesso: {successful} imagens")
        print(f"   Falhas: {failed} imagens")
        
        # Estatísticas
        classes = db.session.query(ImageEmbedding.class_name).distinct().all()
        print(f"   Classes distintas: {len(classes)}")
        
        return True

if __name__ == "__main__":
    # Criar pastas necessárias
    os.makedirs("instance", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)
    
    success = setup_database()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SETUP CONCLUÍDO COM SUCESSO!")
        print("=" * 60)
        print("\nPróximos passos:")
        print("1. Execute: python app_simple.py")
        print("2. Acesse: http://localhost:5000")
        print("3. Faça upload de uma imagem para testar")
    else:
        print("\n❌ SETUP FALHOU")
        print("Verifique se o dataset está na pasta 'epillid/'")
        sys.exit(1)