from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from predictor import PlantCounter
from PIL import Image, ImageDraw, ImageFont
import logging
import base64
from io import BytesIO

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

# Initialize predictor
predictor = PlantCounter('../MODEL_folder/GeCo.pth')

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

@app.route('/get_boxes_image')
def get_boxes_image():
    try:
        # 使用绝对路径
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boxes_output.png')
        if not os.path.exists(image_path):
            logger.error(f'Boxes image not found at: {image_path}')
            return jsonify({'error': 'Boxes image not found'}), 404
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        logger.error(f'Error serving boxes image: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get_mask_image')
def get_mask_image():
    try:
        # 使用绝对路径
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask_output.png')
        if not os.path.exists(image_path):
            logger.error(f'Mask image not found at: {image_path}')
            return jsonify({'error': 'Mask image not found'}), 404
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        logger.error(f'Error serving mask image: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get_labels_image')
def get_labels_image():
    try:
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'labels_output.png')
        if not os.path.exists(image_path):
            logger.error(f'Labels image not found at: {image_path}')
            return jsonify({'error': 'Labels image not found'}), 404
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        logger.error(f'Error serving labels image: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

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
        mask_image, count, pred_boxes = predictor.predict(image, boxes)
        
        # Convert numpy array to PIL Image and save
        if mask_image is not None:
            mask_pil = Image.fromarray(mask_image)
            mask_pil.save('mask_output.png')
            logger.info('Saved mask image')
        
        # Create a blank image with the same size as the input image
        image_with_boxes = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image_with_boxes)
        
        # Draw predicted boxes
        if pred_boxes is not None:
            for box in pred_boxes:
                # Convert from [y1, x1, y2, x2] to [x1, y1, x2, y2]
                x1, y1 = int(box[1]), int(box[0])
                x2, y2 = int(box[3]), int(box[2])
                # Draw rectangle with orange color
                draw.rectangle([y1, x1, y2, x2], outline=(255, 165, 0, 180), width=3)
        
        # Save the image with boxes
        image_with_boxes.save('boxes_output.png')
        logger.info(f'Saved boxes image')

        # Create an image for labels
        labels_image = Image.new('RGBA', image.size, (0, 0, 0, 0)) # Fully transparent background
        draw_labels = ImageDraw.Draw(labels_image)
        try:
            # Try to load a font, fall back to default if not found
            font = ImageFont.truetype("arial.ttf", 20) # Adjust font size as needed
        except IOError:
            logger.warning("Arial font not found, using default font.")
            font = ImageFont.load_default()

        if pred_boxes is not None:
            for i, box in enumerate(pred_boxes):
                x1, y1 = int(box[1]), int(box[0])
                x2, y2 = int(box[3]), int(box[2])

                box_width = x2 - x1
                box_height = y2 - y1

                # Calculate font size based on box dimensions
                # A common heuristic is to use a fraction of the smaller dimension (e.g., 20%-50%)
                # Add a minimum and maximum font size to prevent too small/too large labels
                dynamic_font_size = max(5, min(int(min(box_width, box_height) * 0.4), 60)) # Min 10, Max 60

                try:
                    font = ImageFont.truetype("arial.ttf", dynamic_font_size)
                except IOError:
                    logger.warning("Arial font not found, using default font.")
                    font = ImageFont.load_default()

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                text = str(i + 1)
                # Get accurate text size for centering
                if font == ImageFont.load_default():
                    # For default font, textbbox might not be accurate or available, estimate
                    text_width = draw_labels.textlength(text) # Use textlength if available for default font
                    text_height = font.getsize(text)[1] # getsize is deprecated, but might work for default
                else:
                    left, top, right, bottom = draw_labels.textbbox((0, 0), text, font=font)
                    text_width = right - left
                    text_height = bottom - top

                draw_labels.text((center_y - text_height // 2, center_x - text_width // 2), 
                                 text, 
                                 fill=(255, 255, 0, 255), # Yellow color, fully opaque
                                 font=font)
        
        labels_image.save('labels_output.png')
        logger.info(f'Saved labels image')
        
        logger.info('Prediction completed successfully')
        return jsonify({
            'count': count,
            'boxes': pred_boxes.tolist() if pred_boxes is not None else []
        })
        
    except Exception as e:
        logger.error(f'Error during prediction: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 