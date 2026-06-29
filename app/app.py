import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response
import sys
import os
import io
import csv
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from predict import HateSpeechPredictor

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'), 
            static_folder=os.path.join(BASE_DIR, 'static'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

predictor = None
DATABASE = os.path.join(BASE_DIR, 'hate_speech.db')

def get_db():
    """Returns a SQLite connection with WAL mode enabled for concurrency and row factory configured."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception as e:
        logger.warning(f"Failed to set WAL mode: {e}")
    return conn

def init_db():
    """Initializes the database schema using context managers."""
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME
            )
        ''')
        conn.commit()

def load_system():
    """Pre-loads the deep learning predictor model into memory."""
    global predictor
    try:
        predictor = HateSpeechPredictor(threshold=0.5)
        logger.info("Flask App: Model and Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Flask App: Error loading model artifacts: {e}")
        # In production, we don't want to crash import if artifacts are missing (e.g. during build tasks)
        # but predictor will remain None, throwing a 500 error on /predict which is correct.

# Initialize database and load system on import for WSGI servers
init_db()
load_system()

# ---------------------------------------------------------
# Security Middlewares & Headers
# ---------------------------------------------------------
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response

# ---------------------------------------------------------
# Custom Error Handlers
# ---------------------------------------------------------
@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad Request', 'message': str(e.description)}), 400

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Not Found', 'message': 'The requested URL was not found on the server.'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'error': 'Internal Server Error', 'message': 'An unexpected error occurred on the server.'}), 500

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model not initialized on server.'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide text fields in JSON.'}), 400
        
    text = data['text']
    if not isinstance(text, str):
        return jsonify({'error': 'Input text must be a string.'}), 400
        
    text = text.strip()
    if not text:
        return jsonify({'error': 'Input text cannot be empty.'}), 400
        
    # Input validation limit to avoid long inputs freezing PyTorch models
    if len(text) > 500:
        return jsonify({'error': 'Input text is too long (maximum 500 characters).'}), 400

    result = predictor.predict_single(text)
    
    # Store in database using a secure context manager
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO predictions (text, prediction, confidence, timestamp) VALUES (?, ?, ?, ?)",
                         (text, result['prediction'], float(result['confidence']), datetime.now()))
            conn.commit()
    except Exception as e:
        logger.error(f"Database insertion failed: {e}")
    
    return jsonify({
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities']
    })

@app.route('/analytics', methods=['GET'])
def analytics():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Total counts
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]
            
            # Breakdown by class
            cursor.execute("SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction")
            distribution = dict(cursor.fetchall())
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM predictions")
            avg_conf = cursor.fetchone()[0]
            avg_confidence = round(avg_conf, 4) if avg_conf else 0.0
            
        return jsonify({
            'total_predictions': total_predictions,
            'distribution': distribution,
            'average_confidence': avg_confidence
        })
    except Exception as e:
        logger.error(f"Failed to fetch analytics: {e}")
        return jsonify({'error': 'Database error occurred while fetching analytics.'}), 500

@app.route('/history', methods=['GET', 'DELETE'])
def history():
    if request.method == 'DELETE':
        try:
            with get_db() as conn:
                conn.execute("DELETE FROM predictions")
                conn.commit()
            return jsonify({'message': 'History cleared successfully'})
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return jsonify({'error': 'Database error occurred while clearing history.'}), 500

    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text, prediction, confidence, timestamp FROM predictions ORDER BY id DESC LIMIT 10")
            rows = cursor.fetchall()
        
        history_list = []
        for r in rows:
            history_list.append({
                'text': r['text'],
                'prediction': r['prediction'],
                'confidence': round(r['confidence'], 4),
                'timestamp': r['timestamp']
            })
            
        return jsonify({'history': history_list})
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        return jsonify({'error': 'Database error occurred while fetching history.'}), 500

@app.route('/export', methods=['GET'])
def export_csv():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, text, prediction, confidence, timestamp FROM predictions ORDER BY id DESC")
            rows = cursor.fetchall()

        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['ID', 'Text', 'Prediction', 'Confidence', 'Timestamp'])
        for r in rows:
            cw.writerow([r['id'], r['text'], r['prediction'], r['confidence'], r['timestamp']])
        
        output = si.getvalue()
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=hate_speech_predictions.csv"}
        )
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        return jsonify({'error': 'Database error occurred while exporting predictions.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

