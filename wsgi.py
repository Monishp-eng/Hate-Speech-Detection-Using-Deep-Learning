import os
import sys
import logging
from waitress import serve

# Ensure module path imports work correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from app.app import app

# Setup basic logging for the WSGI server
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('waitress')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Waitress production WSGI server on port {port}...")
    serve(app, host='0.0.0.0', port=port, threads=4)
