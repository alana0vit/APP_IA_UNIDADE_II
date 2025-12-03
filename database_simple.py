from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import base64
import numpy as np

db = SQLAlchemy()

class ImageEmbedding(db.Model):
    """Armazena informações das imagens e seus embeddings"""
    __tablename__ = 'images'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200))
    filepath = db.Column(db.String(500))
    class_name = db.Column(db.String(100))
    embedding = db.Column(db.PickleType)  # Usando PickleType diretamente
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'filepath': self.filepath,
            'class_name': self.class_name
        }

class Search(db.Model):
    """Histórico de buscas"""
    __tablename__ = 'searches'
    
    id = db.Column(db.Integer, primary_key=True)
    query_image = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    results_count = db.Column(db.Integer)

def init_db(app):
    """Inicializa o banco de dados"""
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
    
    return db