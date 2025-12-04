from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
from model_loader import ImageClassifier
from database import db, UploadedImage, SearchResult, init_db

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'sua-chave-secreta-aqui')

# Configurações
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Criar pastas necessárias
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Inicializar banco de dados
init_db(app)

# Inicializar classificador de imagens (pode demorar na primeira execução)
print("Carregando modelo CLIP...")
classifier = ImageClassifier()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
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
            'image_id': uploaded_image.id
        })
    
    return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

@app.route('/search', methods=['POST'])
def search_similar():
    data = request.json
    filename = data.get('filename')
    k = data.get('k', 5)
    
    if not filename:
        return jsonify({'error': 'Nome do arquivo não fornecido'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Arquivo não encontrado'}), 404
    
    # Buscar imagens similares
    results = classifier.search_similar_images(filepath, k=k)
    
    # Salvar resultados no banco de dados
    uploaded_image = UploadedImage.query.filter_by(filename=filename).first()
    
    if uploaded_image:
        for result in results:
            search_result = SearchResult(
                uploaded_image_id=uploaded_image.id,
                similar_image_path=result['path'],
                similarity_score=result['distance']
            )
            db.session.add(search_result)
        db.session.commit()
    
    # Preparar resposta
    response_results = []
    for result in results:
        # Converter caminho relativo para URL
        rel_path = os.path.relpath(result['path'], start='.')
        response_results.append({
            'path': rel_path,
            'distance': result['distance'],
            'filename': result['filename']
        })
    
    return jsonify({
        'success': True,
        'results': response_results,
        'count': len(response_results)
    })

@app.route('/history')
def history():
    # Obter histórico de uploads
    uploads = UploadedImage.query.order_by(UploadedImage.upload_date.desc()).limit(50).all()
    return render_template('history.html', uploads=uploads)

@app.route('/api/history')
def api_history():
    uploads = UploadedImage.query.order_by(UploadedImage.upload_date.desc()).limit(50).all()
    return jsonify([upload.to_dict() for upload in uploads])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/dataset/<path:filename>')
def dataset_file(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)