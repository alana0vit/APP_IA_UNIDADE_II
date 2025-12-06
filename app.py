from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import json
from model_loader import ImageClassifier
from database import db, UploadedImage, SearchResult, init_db

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'sua-chave-secreta-aqui')

# Configurações
UPLOAD_FOLDER = 'uploads'
EMBEDDINGS_FOLDER = 'embeddings'
SAMPLE_IMAGES_FOLDER = 'sample_images'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max

# Criar pastas necessárias
for folder in [UPLOAD_FOLDER, EMBEDDINGS_FOLDER, SAMPLE_IMAGES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Inicializar banco de dados
init_db(app)

# Inicializar classificador de imagens
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Página inicial"""
    sample_images = []
    stats = {}
    
    if classifier:
        sample_images = classifier.get_sample_images(12)
        stats = classifier.get_dataset_stats()
    
    return render_template('index.html', 
                          sample_images=sample_images, 
                          stats=stats,
                          model_loaded=classifier is not None)

@app.route('/api/stats')
def get_stats():
    """API para obter estatísticas"""
    if not classifier:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    stats = classifier.get_dataset_stats()
    return jsonify(stats)

@app.route('/upload', methods=['POST'])
def upload_image():
    if not classifier:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
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
        file.save(filepath)
        file_size = os.path.getsize(filepath)
        
        # Verificar se a imagem é válida
        try:
            from PIL import Image
            Image.open(filepath).verify()
        except:
            os.remove(filepath)
            return jsonify({'error': 'Arquivo de imagem inválido'}), 400
        
        # Salvar no banco de dados
        uploaded_image = UploadedImage(
            filename=filename,
            original_filename=original_filename,
            file_size=file_size
        )
        db.session.add(uploaded_image)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'original_filename': original_filename,
            'image_id': uploaded_image.id,
            'file_size': file_size
        })
    
    return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

@app.route('/search', methods=['POST'])
def search_similar():
    if not classifier:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    data = request.json
    filename = data.get('filename')
    k = min(data.get('k', 5), 10)  # Limitar a 10 resultados máximo
    
    if not filename:
        return jsonify({'error': 'Nome do arquivo não fornecido'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Arquivo não encontrado'}), 404
    
    # Buscar imagens similares
    results = classifier.search_similar_images(filepath, k=k)
    
    # Salvar resultados no banco de dados
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
    
    # Preparar resposta
    response_results = []
    for result in results:
        # Converter caminho para URL acessível
        display_path = result['display_path']
        if os.path.exists(display_path):
            # Se estiver na pasta de samples
            if display_path.startswith(SAMPLE_IMAGES_FOLDER):
                url_path = f"/{display_path}"
            else:
                # Tentar servir do caminho original
                url_path = f"/dataset/{os.path.relpath(display_path, start='.')}"
        else:
            continue  # Pular se não existir
        
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
        'upload_id': uploaded_image.id if uploaded_image else None
    })

@app.route('/history')
def history():
    """Página de histórico"""
    uploads = UploadedImage.query.order_by(UploadedImage.upload_date.desc()).limit(50).all()
    
    # Adicionar contagem de resultados para cada upload
    for upload in uploads:
        upload.result_count = SearchResult.query.filter_by(uploaded_image_id=upload.id).count()
    
    return render_template('history.html', uploads=uploads)

@app.route('/api/history')
def api_history():
    """API para histórico"""
    uploads = UploadedImage.query.order_by(UploadedImage.upload_date.desc()).limit(50).all()
    
    result = []
    for upload in uploads:
        upload_dict = upload.to_dict()
        upload_dict['result_count'] = SearchResult.query.filter_by(uploaded_image_id=upload.id).count()
        result.append(upload_dict)
    
    return jsonify(result)

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
        if os.path.exists(filename):
            return send_from_directory('.', filename)
        else:
            # Tentar encontrar na pasta de samples
            basename = os.path.basename(filename)
            sample_path = os.path.join(SAMPLE_IMAGES_FOLDER, basename)
            if os.path.exists(sample_path):
                return send_from_directory(SAMPLE_IMAGES_FOLDER, basename)
            else:
                return "Arquivo não encontrado", 404
    except:
        return "Erro ao carregar arquivo", 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Limpa histórico antigo"""
    try:
        # Manter apenas os últimos 100 uploads
        old_uploads = UploadedImage.query.order_by(UploadedImage.upload_date.desc()).offset(100).all()
        
        for upload in old_uploads:
            # Deletar resultados relacionados
            SearchResult.query.filter_by(uploaded_image_id=upload.id).delete()
            # Deletar arquivo físico
            filepath = os.path.join(UPLOAD_FOLDER, upload.filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            # Deletar do banco
            db.session.delete(upload)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'deleted': len(old_uploads)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)