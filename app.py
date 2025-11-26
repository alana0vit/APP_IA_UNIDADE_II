import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
import datetime
import pandas as pd
import os

# Configuração da página
st.set_page_config(
    page_title="Classificador de Pílulas",
    page_icon="💊",
    layout="wide"
)

# Inicializar banco de dados
def init_db():
    conn = sqlite3.connect('database/interactions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            predicted_class INTEGER,
            confidence REAL,
            image_size TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Carregar modelo
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model/advanced_pill_classification.keras')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

# Preprocessamento da imagem
def preprocess_image(image):
    image = image.resize((64, 64))
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Salvar predição no banco
def save_prediction(filename, predicted_class, confidence, image_size):
    conn = sqlite3.connect('database/interactions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (timestamp, filename, predicted_class, confidence, image_size)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        datetime.datetime.now().isoformat(),
        filename,
        int(predicted_class),
        float(confidence),
        str(image_size)
    ))
    conn.commit()
    conn.close()

# Interface principal
def main():
    st.title("💊 Classificador Inteligente de Pílulas")
    st.markdown("""
    Este sistema utiliza inteligência artificial para classificar pílulas baseado em características visuais como cor e formato.
    Faça upload de uma imagem e veja a classificação!
    """)
    
    # Sidebar
    st.sidebar.title("Sobre o Projeto")
    st.sidebar.info("""
    **Dataset**: Pill Images from Kaggle  
    **Modelo**: CNN Avançada  
    **Classes**: 8 grupos por cor/formato  
    **Precisão**: ~85% nos testes
    """)
    
    # Carregar modelo
    model = load_model()
    
    if model is None:
        st.error("Modelo não encontrado. Verifique se o arquivo do modelo está na pasta 'model/'")
        return
    
    # Upload de imagem
    uploaded_file = st.file_uploader(
        "Escolha uma imagem de pílula",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos suportados: JPG, JPEG, PNG"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            # Mostrar imagem
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem Carregada", use_column_width=True)
            
            # Processar imagem
            processed_image = preprocess_image(image)
            
            # Fazer predição
            if st.button("🔍 Classificar Pílula"):
                with st.spinner("Analisando imagem..."):
                    predictions = model.predict(processed_image, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])
                    
                    # Mapeamento de classes (ajuste conforme seu modelo)
                    class_names = {
                        0: "Branco/Claro",
                        1: "Vermelho/Alaranjado", 
                        2: "Verde",
                        3: "Azul",
                        4: "Amarelo",
                        5: "Roxo/Magenta",
                        6: "Ciano/Azul Claro",
                        7: "Outras Cores"
                    }
                    
                    # Resultados
                    st.success("✅ Classificação Concluída!")
                    
                    # Mostrar resultados
                    st.subheader("Resultados:")
                    st.write(f"**Classe Predita**: {class_names.get(predicted_class, 'Desconhecida')}")
                    st.write(f"**Confiança**: {confidence:.2%}")
                    st.write(f"**Classe ID**: {predicted_class}")
                    
                    # Barra de confiança
                    st.progress(float(confidence))
                    
                    # Salvar no banco
                    save_prediction(
                        uploaded_file.name,
                        predicted_class,
                        confidence,
                        image.size
                    )
                    
                    st.balloons()
    
    with col2:
        st.subheader("📊 Estatísticas do Sistema")
        
        # Mostrar estatísticas do banco
        try:
            conn = sqlite3.connect('database/interactions.db')
            df = pd.read_sql('SELECT * FROM predictions', conn)
            conn.close()
            
            if not df.empty:
                st.metric("Total de Classificações", len(df))
                st.metric("Última Classificação", df.iloc[-1]['timestamp'][:16])
                
                # Distribuição de classes
                class_dist = df['predicted_class'].value_counts()
                st.bar_chart(class_dist)
            else:
                st.info("Nenhuma classificação registrada ainda.")
                
        except Exception as e:
            st.info("Banco de dados inicializado. Aguardando primeira classificação.")
    
    # Seção de histórico
    st.subheader("📈 Histórico de Classificações")
    try:
        conn = sqlite3.connect('database/interactions.db')
        df = pd.read_sql('''
            SELECT timestamp, filename, predicted_class, confidence 
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''', conn)
        conn.close()
        
        if not df.empty:
            st.dataframe(df)
        else:
            st.info("Nenhuma classificação no histórico.")
    except:
        st.info("Histórico não disponível.")

# Inicializar app
if __name__ == "__main__":
    init_db()
    main()