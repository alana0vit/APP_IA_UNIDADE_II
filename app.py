import os
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
import matplotlib
matplotlib.use('Agg')  # Usar backend não interativo
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta_aqui'  # Mude para produção!
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Variáveis globais para o modelo
device = None
model = None
preprocess = None
index = None
image_paths = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def setup_model():
    """Configura o modelo CLIP e carrega as embeddings"""
    global device, model, preprocess, index, image_paths
    
    try:
        print("Iniciando setup do modelo...")
        
        # Configurar dispositivo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Dispositivo: {device}")
        
        # Carregar modelo CLIP
        print("Carregando modelo CLIP...")
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Verificar se as embeddings existem
        embeddings_path = "model/embeddings.npy"
        image_paths_file = "model/image_paths.npy"
        
        if os.path.exists(embeddings_path) and os.path.exists(image_paths_file):
            print("Carregando embeddings pré-calculadas...")
            embeddings = np.load(embeddings_path)
            image_paths = np.load(image_paths_file, allow_pickle=True).tolist()
        else:
            print("Embeddings não encontradas. Execute primeiro o setup_model.py")
            return False
        
        # Criar índice FAISS
        print("Criando índice FAISS...")
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.astype("float32"))
        
        print("Setup do modelo concluído com sucesso!")
        return True
        
    except Exception as e:
        print(f"Erro no setup do modelo: {str(e)}")
        return False

def similar(image_path, k=5):
    """Encontra imagens similares"""
    try:
        # Carregar e processar imagem
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        # Gerar embedding da imagem de consulta
        with torch.no_grad():
            query_emb = model.encode_image(img_tensor)
            query_emb /= query_emb.norm()
            query_emb = query_emb.cpu().numpy().astype("float32")
        
        # Buscar no índice FAISS
        distances, indices = index.search(query_emb, k)
        
        # Preparar resultados
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(image_paths):  # Verificar índice válido
                result = {
                    'rank': i + 1,
                    'distance': float(dist),
                    'path': image_paths[idx],
                    'filename': os.path.basename(image_paths[idx])
                }
                results.append(result)
        
        return results, None
        
    except Exception as e:
        return None, str(e)

def plot_results(query_img_path, results):
    """Cria um gráfico com os resultados"""
    try:
        # Criar figura com subplots
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(4 * (len(results) + 1), 4))
        
        # Plotar imagem de consulta
        query_img = Image.open(query_img_path)
        axes[0].imshow(query_img)
        axes[0].set_title('Imagem de Consulta')
        axes[0].axis('off')
        
        # Plotar imagens similares
        for i, result in enumerate(results):
            try:
                img = Image.open(result['path'])
                axes[i + 1].imshow(img)
                axes[i + 1].set_title(f'Similar {i+1}\nDist: {result["distance"]:.3f}')
                axes[i + 1].axis('off')
            except:
                axes[i + 1].text(0.5, 0.5, 'Imagem não encontrada', 
                                ha='center', va='center')
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        # Converter para base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Erro ao criar gráfico: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Verificar se o arquivo foi enviado
        if 'file' not in request.files:
            flash('Nenhum arquivo selecionado')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('Nenhum arquivo selecionado')
            return redirect(request.url)
        
        if file and allowed_file(file.name):
            # Salvar arquivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Obter número de resultados
            k = int(request.form.get('k', 5))
            
            # Buscar imagens similares
            results, error = similar(filepath, k=k)
            
            if error:
                flash(f'Erro ao processar imagem: {error}')
                return redirect(request.url)
            
            # Gerar visualização
            plot_url = plot_results(filepath, results[:min(k, len(results))])
            
            return render_template('results.html', 
                                 query_image=filepath,
                                 results=results,
                                 plot_url=plot_url,
                                 k=k)
        else:
            flash('Tipo de arquivo não permitido. Use PNG, JPG ou JPEG')
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index.html', section='about')

if __name__ == '__main__':
    # Criar diretórios necessários
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # Configurar modelo
    print("Inicializando aplicação...")
    if setup_model():
        print("Modelo carregado com sucesso!")
        # Para produção no Render, usar host='0.0.0.0' e port adequada
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("Falha ao carregar o modelo. Verifique se as embeddings foram geradas.")