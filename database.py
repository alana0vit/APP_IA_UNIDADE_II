from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

db = SQLAlchemy()

class UploadedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    file_size = db.Column(db.Integer)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'upload_date': self.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
            'file_size': self.file_size
        }

class SearchResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uploaded_image_id = db.Column(db.Integer, db.ForeignKey('uploaded_image.id'), nullable=False)
    similar_image_path = db.Column(db.String(500), nullable=False)
    similarity_score = db.Column(db.Float)
    search_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    uploaded_image = db.relationship('UploadedImage', backref='search_results')

def init_db(app):
    """Inicializa o banco de dados"""
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "database.db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()