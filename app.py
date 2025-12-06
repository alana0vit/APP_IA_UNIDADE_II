import sys
import subprocess
import pkg_resources
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import json

# Verificar dependências ao iniciar
REQUIRED_PACKAGES = [
    'Flask>=2.3.2',
    'Pillow>=9.5.0',
    'torch>=1.13.1',
    'faiss-cpu>=1.7.4',
    'numpy>=1.24.3'
]

def check_dependencies():
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            pkg_name = package.split('>=')[0].split('==')[0]
            pkg_resources.require(package)
            print(f"✅ {pkg_name} está instalado")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(pkg_name)
        except pkg_resources.VersionConflict as e:
            print(f"⚠️  Versão de {package}: {e}")
    
    if missing_packages:
        print(f"❌ Pacotes faltando: {missing_packages}")
        print("Instalando pacotes faltantes...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} instalado")
            except subprocess.CalledProcessError as e:
                print(f"❌ Falha ao instalar {package}: {e}")
    
    return len(missing_packages) == 0

# Verificar dependências
print("Verificando dependências...")
check_dependencies()

# Agora importar o model_loader
try:
    from model_loader import ImageClassifier
    MODEL_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Erro ao importar model_loader: {e}")
    MODEL_LOADER_AVAILABLE = False

# Importar database
try:
    from database import db, UploadedImage, SearchResult, init_db
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Erro ao importar database: {e}")
    DATABASE_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configurações
UPLOAD_FOLDER = 'uploads'
EMBEDDINGS_FOLDER = 'embeddings'
SAMPLE_IMAGES_FOLDER = 'sample_images'
STATIC_FOLDER = 'static'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max

# Criar pastas necessárias
for folder in [UPLOAD_FOLDER, EMBEDDINGS_FOLDER, SAMPLE_IMAGES_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Subpastas do static
os.makedirs(os.path.join(STATIC_FOLDER, 'css'), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'js'), exist_ok=True)

# Inicializar banco de dados
if DATABASE_AVAILABLE:
    try:
        init_db(app)
        print("✅ Banco de dados inicializado")
    except Exception as e:
        print(f"❌ Erro ao inicializar banco de dados: {e}")
        DATABASE_AVAILABLE = False

# Inicializar classificador de imagens
classifier = None
if MODEL_LOADER_AVAILABLE:
    print("Carregando modelo CLIP e embeddings...")
    try:
        classifier = ImageClassifier(
            embeddings_path=EMBEDDINGS_FOLDER,
            sample_images_path=SAMPLE_IMAGES_FOLDER
        )
        print("✅ Modelo carregado com sucesso!")
        
        # Verificar estatísticas
        stats = classifier.get_dataset_stats()
        print(f"📊 Estatísticas do dataset:")
        print(f"   - Imagens no índice: {stats['total_images']}")
        print(f"   - Dimensões dos embeddings: {stats['embedding_dimensions']}")
        print(f"   - Tem imagens de exemplo: {stats['has_sample_images']}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        classifier = None
else:
    print("⚠️  Model loader não disponível")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Página inicial"""
    sample_images = []
    stats = {}
    model_loaded = classifier is not None
    
    if classifier:
        try:
            sample_images = classifier.get_sample_images(12)
            stats = classifier.get_dataset_stats()
        except Exception as e:
            print(f"Erro ao obter amostras/estatísticas: {e}")
    
    return render_template('index.html', 
                          sample_images=sample_images, 
                          stats=stats,
                          model_loaded=model_loaded)

@app.route('/api/stats')
def get_stats():
    """API para obter estatísticas"""
    if not classifier:
        return jsonify({'error': 'Modelo não carregado', 'model_loaded': False}), 500
    
    try:
        stats = classifier.get_dataset_stats()
        stats['model_loaded'] = True
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e), 'model_loaded': False}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    if not classifier:
        return jsonify({'error': 'Modelo não carregado', 'model_loaded': False}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if file and allowed_file(file.filename):
        # Gerar nome único para o arquivo
        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4().hex}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Salvar arquivo
        try:
            file.save(filepath)
            file_size = os.path.getsize(filepath)
            
            # Verificar se a imagem é válida
            try:
                from PIL import Image
                img = Image.open(filepath)
                img.verify()  # Verificar integridade
                img = Image.open(filepath)  # Reabrir para uso
                img_format = img.format
                img_size = img.size
            except Exception as img_error:
                os.remove(filepath)
                return jsonify({'error': f'Arquivo de imagem inválido: {img_error}'}), 400
            
            # Salvar no banco de dados
            uploaded_image = None
            if DATABASE_AVAILABLE:
                try:
                    uploaded_image = UploadedImage(
                        filename=filename,
                        original_filename=original_filename,
                        file_size=file_size
                    )
                    db.session.add(uploaded_image)
                    db.session.commit()
                    image_id = uploaded_image.id
                except Exception as db_error:
                    print(f"Erro ao salvar no banco: {db_error}")
                    image_id = None
            else:
                image_id = None
            
            return jsonify({
                'success': True,
                'filename': filename,
                'original_filename': original_filename,
                'image_id': image_id,
                'file_size': file_size,
                'image_format': img_format,
                'image_dimensions': img_size,
                'model_loaded': True
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Erro ao processar arquivo: {str(e)}'}), 500
    
    return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

@app.route('/search', methods=['POST'])
def search_similar():
    if not classifier:
        return jsonify({'error': 'Modelo não carregado', 'model_loaded': False}), 500
    
    data = request.json
    filename = data.get('filename')
    k = min(int(data.get('k', 5)), 10)  # Limitar a 10 resultados máximo
    
    if not filename:
        return jsonify({'error': 'Nome do arquivo não fornecido'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Arquivo não encontrado'}), 404
    
    try:
        # Buscar imagens similares
        results = classifier.search_similar_images(filepath, k=k)
        
        # Salvar resultados no banco de dados
        uploaded_image = None
        if DATABASE_AVAILABLE:
            try:
                uploaded_image = UploadedImage.query.filter_by(filename=filename).first()
                if uploaded_image and results:
                    for result in results:
                        search_result = SearchResult(
                            uploaded_image_id=uploaded_image.id,
                            similar_image_path=result['original_path'],
                            similarity_score=result['distance']
                        )
                        db.session.add(search_result)
                    db.session.commit()
            except Exception as db_error:
                print(f"Erro ao salvar resultados: {db_error}")
        
        # Preparar resposta
        response_results = []
        for result in results:
            # Converter caminho para URL acessível
            display_path = result['display_path']
            url_path = ""
            
            if os.path.exists(display_path):
                # Se estiver na pasta de samples
                if display_path.startswith(SAMPLE_IMAGES_FOLDER):
                    url_path = f"/sample_images/{os.path.basename(display_path)}"
                else:
                    # Tentar servir do caminho original
                    try:
                        rel_path = os.path.relpath(display_path, start='.')
                        url_path = f"/dataset/{rel_path}"
                    except:
                        url_path = ""
            else:
                # Se o arquivo não existir localmente, pular
                continue
            
            response_results.append({
                'url': url_path,
                'original_path': result['original_path'],
                'distance': result['distance'],
                'similarity_percent': round(result['similarity_percent'], 1),
                'filename': result['filename']
            })
        
        return jsonify({
            'success': True,
            'results': response_results,
            'count': len(response_results),
            'upload_id': uploaded_image.id if uploaded_image else None,
            'model_loaded': True
        })
        
    except Exception as e:
        print(f"Erro na busca: {e}")
        return jsonify({'error': f'Erro na busca: {str(e)}', 'model_loaded': True}), 500

@app.route('/history')
def history():
    """Página de histórico"""
    uploads = []
    if DATABASE_AVAILABLE:
        try:
            uploads = UploadedImage.query.order_by(UploadedImage.upload_date.desc()).limit(50).all()
            
            # Adicionar contagem de resultados para cada upload
            for upload in uploads:
                upload.result_count = SearchResult.query.filter_by(uploaded_image_id=upload.id).count()
        except Exception as e:
            print(f"Erro ao carregar histórico: {e}")
    
    return render_template('history.html', uploads=uploads, database_available=DATABASE_AVAILABLE)

@app.route('/api/history')
def api_history():
    """API para histórico"""
    uploads_data = []
    
    if DATABASE_AVAILABLE:
        try:
            uploads = UploadedImage.query.order_by(UploadedImage.upload_date.desc()).limit(50).all()
            
            for upload in uploads:
                upload_dict = {
                    'id': upload.id,
                    'filename': upload.filename,
                    'original_filename': upload.original_filename,
                    'upload_date': upload.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'file_size': upload.file_size
                }
                upload_dict['result_count'] = SearchResult.query.filter_by(uploaded_image_id=upload.id).count()
                uploads_data.append(upload_dict)
        except Exception as e:
            print(f"Erro na API de histórico: {e}")
    
    return jsonify({
        'success': True,
        'uploads': uploads_data,
        'database_available': DATABASE_AVAILABLE
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/sample_images/<filename>')
def sample_image(filename):
    return send_from_directory(SAMPLE_IMAGES_FOLDER, filename)

@app.route('/dataset/<path:filename>')
def dataset_file(filename):
    """Serve arquivos do dataset original se existirem"""
    try:
        # Verificar se o arquivo existe
        if os.path.exists(filename):
            return send_from_directory('.', filename)
        
        # Tentar encontrar na pasta de samples
        basename = os.path.basename(filename)
        sample_path = os.path.join(SAMPLE_IMAGES_FOLDER, basename)
        if os.path.exists(sample_path):
            return send_from_directory(SAMPLE_IMAGES_FOLDER, basename)
        
        # Se não encontrar, retornar imagem placeholder
        return "Arquivo não encontrado", 404
        
    except Exception as e:
        print(f"Erro ao servir arquivo {filename}: {e}")
        return "Erro ao carregar arquivo", 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Limpa histórico antigo"""
    if not DATABASE_AVAILABLE:
        return jsonify({'error': 'Banco de dados não disponível'}), 500
    
    try:
        # Manter apenas os últimos 50 uploads
        all_uploads = UploadedImage.query.order_by(UploadedImage.upload_date.desc()).all()
        
        if len(all_uploads) > 50:
            old_uploads = all_uploads[50:]
            deleted_count = 0
            
            for upload in old_uploads:
                # Deletar resultados relacionados
                SearchResult.query.filter_by(uploaded_image_id=upload.id).delete()
                
                # Deletar arquivo físico
                filepath = os.path.join(UPLOAD_FOLDER, upload.filename)
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                
                # Deletar do banco
                db.session.delete(upload)
                deleted_count += 1
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Histórico limpo: {deleted_count} registros removidos',
                'deleted': deleted_count
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Nada para limpar (menos de 50 registros)',
                'deleted': 0
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Endpoint de verificação de saúde da aplicação"""
    health_status = {
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'database_available': DATABASE_AVAILABLE,
        'model_loader_available': MODEL_LOADER_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    }
    
    # Verificar pastas essenciais
    essential_folders = [UPLOAD_FOLDER, EMBEDDINGS_FOLDER, SAMPLE_IMAGES_FOLDER]
    for folder in essential_folders:
        health_status[f'folder_{folder}_exists'] = os.path.exists(folder)
    
    # Verificar arquivos de embeddings
    embeddings_files = ['embeddings.npy', 'image_paths.npy']
    for file in embeddings_files:
        filepath = os.path.join(EMBEDDINGS_FOLDER, file)
        health_status[f'file_{file}_exists'] = os.path.exists(filepath)
    
    return jsonify(health_status)

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve arquivos estáticos"""
    return send_from_directory(STATIC_FOLDER, filename)

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Página não encontrada'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Erro interno do servidor'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'Arquivo muito grande. Tamanho máximo: 8MB'}), 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"\n🚀 Iniciando aplicação...")
    print(f"   Porta: {port}")
    print(f"   Modo debug: {debug_mode}")
    print(f"   Modelo carregado: {classifier is not None}")
    print(f"   Banco de dados: {DATABASE_AVAILABLE}")
    
    # Verificar se temos os arquivos necessários
    embeddings_file = os.path.join(EMBEDDINGS_FOLDER, 'embeddings.npy')
    if not os.path.exists(embeddings_file):
        print(f"⚠️  AVISO: Arquivo de embeddings não encontrado em {embeddings_file}")
        print("   A aplicação funcionará, mas não poderá fazer buscas.")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)