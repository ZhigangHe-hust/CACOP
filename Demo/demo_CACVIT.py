from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from predictor import PlantCounter, convert_np_image_to_base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize 
predictor = PlantCounter('../PlantCount_CACViT/checkpoints/best_model.pth')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def demo():
    return render_template('demo.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error('No file part in request')
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f'Saving file to: {filepath}')
        file.save(filepath)
        return jsonify({'filename': filename})
    
    logger.error(f'Invalid file type: {file.filename}')
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    filename = data.get('filename')
    boxes = data.get('boxes', [])
    
    if not filename or not boxes:
        logger.error('Missing filename or boxes')
        return jsonify({'error': 'Missing filename or boxes'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    logger.info(f'Loading file from: {filepath}')
    
    if not os.path.exists(filepath):
        logger.error(f'File not found: {filepath}')
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Load image and get prediction
        logger.info('Opening image file')
        image = Image.open(filepath)
        logger.info('Getting prediction')
        cluster_image_np, count, density_map_np = predictor.predict(image, boxes)
        
        # Convert cluster image to base64
        buffer = BytesIO()
        Image.fromarray(cluster_image_np).save(buffer, format='PNG')
        buffer.seek(0)
        cluster_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Convert original density map to base64 with colormap
        density_map_original_base64 = convert_np_image_to_base64(density_map_np)
        
        logger.info('Prediction completed successfully')
        return jsonify({
            'count': float(count),
            'density_map': f'data:image/png;base64,{cluster_image_base64}',
            'density_map_original': f'data:image/png;base64,{density_map_original_base64}'
        })
        
    except Exception as e:
        logger.error(f'Error during prediction: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 