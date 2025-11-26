import os
import sqlite3

def create_directories():
    """Cria a estrutura de diretórios do projeto"""
    directories = ['database', 'model', 'utils', 'assets']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Diretório '{directory}' criado/verificado")

def init_database():
    """Inicializa o banco de dados SQLite"""
    conn = sqlite3.connect('database/interactions.db')
    c = conn.cursor()
    
    # Tabela para armazenar as predições
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
    
    # Tabela para estatísticas de uso
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            total_predictions INTEGER,
            avg_confidence REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Banco de dados inicializado")

def check_model_file():
    """Verifica se o arquivo do modelo existe"""
    model_path = 'model/advanced_pill_classification.keras'
    if os.path.exists(model_path):
        print("✅ Arquivo do modelo encontrado")
        return True
    else:
        print("⚠️  Arquivo do modelo NÃO encontrado")
        print("   Por favor, adicione o arquivo 'advanced_pill_classification.keras' na pasta 'model/'")
        return False

def main():
    """Função principal do setup"""
    print("🎯 Iniciando setup do projeto...")
    print("=" * 50)
    
    create_directories()
    init_database()
    model_exists = check_model_file()
    
    print("=" * 50)
    if model_exists:
        print("🎉 Setup concluído com SUCESSO!")
        print("\n📝 Próximos passos:")
        print("   1. Execute: streamlit run app.py")
        print("   2. Acesse: http://localhost:8501")
        print("   3. Teste o upload de uma imagem")
    else:
        print("⚠️  Setup parcialmente concluído")
        print("   Adicione o modelo na pasta 'model/' antes de executar o app")

if __name__ == "__main__":
    main()