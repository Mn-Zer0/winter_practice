from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()


class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    violation_detected = db.Column(db.Boolean, nullable=False)
    violation_ratio = db.Column(db.Float)
    total_motorcycles = db.Column(db.Integer)
    processing_time = db.Column(db.Float)
    bounding_boxes = db.Column(db.Text)  # JSON строка с координатами
    image_path = db.Column(db.String(300))

    def to_dict(self):
        """Конвертация в словарь для JSON/Excel"""
        return {
            'id': self.id,
            'filename': self.filename,
            'timestamp': self.upload_time.isoformat(),
            'violation': self.violation_detected,
            'violation_ratio': self.violation_ratio,
            'total_motorcycles': self.total_motorcycles,
            'processing_time_seconds': self.processing_time,
            'image_path': self.image_path
        }

    def get_bounding_boxes(self):
        """Получение bounding boxes из JSON"""
        if self.bounding_boxes:
            return json.loads(self.bounding_boxes)
        return []