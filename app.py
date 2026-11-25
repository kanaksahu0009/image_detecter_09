"""
AI Image Detection Web Application
====================================
Flask server providing REST API and interactive UI for detecting
AI-generated vs authentic human images with probability breakdown.
"""

import os
import io
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

from detector import AIDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['SAMPLES_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'samples')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAMPLES_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}

detector = AIDetector()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect_image():
    """Endpoint for uploading and analyzing an image."""
    start_time = time.time()

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': f'Unsupported file format. Please upload one of: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        # Read image in-memory
        file_bytes = file.read()
        pil_img = Image.open(io.BytesIO(file_bytes))
        
        # Execute detection pipeline
        result = detector.detect(pil_img)
        
        elapsed = round(time.time() - start_time, 2)
        result['elapsed_seconds'] = elapsed
        result['filename'] = secure_filename(file.filename)
        result['file_size_kb'] = round(len(file_bytes) / 1024.0, 1)

        return jsonify(result)

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/sample/<sample_name>', methods=['GET'])
def detect_sample(sample_name):
    """Analyze a bundled sample image for quick demonstration."""
    sample_path = os.path.join(app.config['SAMPLES_FOLDER'], f"{sample_name}.jpg")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(app.config['SAMPLES_FOLDER'], f"{sample_name}.png")
    
    if not os.path.exists(sample_path):
        return jsonify({'status': 'error', 'message': 'Sample image not found'}), 404

    try:
        start_time = time.time()
        with open(sample_path, 'rb') as f:
            file_bytes = f.read()
        pil_img = Image.open(io.BytesIO(file_bytes))

        result = detector.detect(pil_img)
        elapsed = round(time.time() - start_time, 2)
        result['elapsed_seconds'] = elapsed
        result['filename'] = os.path.basename(sample_path)
        result['file_size_kb'] = round(len(file_bytes) / 1024.0, 1)

        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Sample test failed: {str(e)}'}), 500


@app.route('/static/samples/<filename>')
def serve_sample(filename):
    """Serves sample image files."""
    return send_from_directory(app.config['SAMPLES_FOLDER'], filename)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("   AI Generated Image Detector & Probability Analyzer")
    print("   Ready at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=True)
