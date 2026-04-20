import os
import torch
import numpy as np
from PIL import Image, ImageOps
import bchlib
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import base64
from threading import Lock
import logging

# Add paths for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'StegaStamp-pytorch'))
from stegastamp.models import StegaStampDecoder

# Configuration
UPLOAD_FOLDER = '/tmp/image_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
BCH_POLYNOMIAL = 137
BCH_BITS = 5
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = None
decoder_lock = Lock()
bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)

# Decoder configuration
DECODER_CONFIG = {
    'secret_size': 100,
    'height': 400,
    'width': 400,
    'model_path': './StegaStamp-pytorch/asset/best.pth'
}


def bits_to_bytes(bits):
    """Convert bit array to bytes."""
    out = bytearray()
    for b in range(0, len(bits), 8):
        byte = 0
        for i in range(8):
            if b + i < len(bits):
                byte = (byte << 1) | bits[b + i]
        out.append(byte)
    return out


def initialize_decoder():
    """Initialize the StegaStamp decoder model."""
    global decoder
    if decoder is not None:
        return
    
    try:
        logger.info(f"Initializing decoder on device: {device}")
        decoder = StegaStampDecoder(
            secret_size=DECODER_CONFIG['secret_size'],
            height=DECODER_CONFIG['height'],
            width=DECODER_CONFIG['width']
        ).to(device)
        
        model_path = DECODER_CONFIG['model_path']
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "decoder" in ckpt:
                decoder.load_state_dict(ckpt["decoder"])
            else:
                decoder.load_state_dict(ckpt)
        else:
            logger.warning(f"Model not found at {model_path}. Using untrained model.")
        
        decoder.eval()
        logger.info("Decoder initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize decoder: {e}")
        raise


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_image(image_path_or_pil, return_confidence=True):
    """
    Decode a single image and return decoded message and confidence.
    
    Args:
        image_path_or_pil: File path or PIL Image
        return_confidence: Whether to return confidence metric
    
    Returns:
        dict with 'success', 'message', 'confidence', 'error'
    """
    try:
        with decoder_lock:
            if decoder is None:
                initialize_decoder()
            
            # Load image
            if isinstance(image_path_or_pil, str):
                image = Image.open(image_path_or_pil).convert("RGB")
            else:
                image = image_path_or_pil.convert("RGB")
            
            # Resize image
            size = (DECODER_CONFIG['width'], DECODER_CONFIG['height'])
            image = ImageOps.fit(image, size)
            
            # Convert to tensor
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Decode
            with torch.no_grad():
                logits = decoder(image_tensor)
                logits_cpu = logits.cpu()
                
                # Calculate confidence as average sigmoid probability
                sigmoid_probs = torch.sigmoid(logits_cpu)
                confidence = float(torch.mean(sigmoid_probs).item())
                
                # Convert logits to bits
                bits = torch.round(sigmoid_probs).squeeze(0).numpy().astype(np.uint8).tolist()
            
            # BCH error correction and decoding
            raw = bits_to_bytes(bits[:-4])
            data, ecc = raw[:7], raw[7:]
            
            try:
                bch.decode(bytearray(data), bytearray(ecc))
                decoded_message = bytes(data).decode('utf-8').strip()
                
                return {
                    'success': True,
                    'message': decoded_message,
                    'confidence': confidence,
                    'error': None
                }
            except Exception as e:
                return {
                    'success': False,
                    'message': None,
                    'confidence': confidence,
                    'error': f'BCH decoding failed: {str(e)}'
                }
    
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return {
            'success': False,
            'message': None,
            'confidence': 0.0,
            'error': str(e)
        }


def image_to_base64(image_path_or_pil):
    """Convert image to base64 for display."""
    try:
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil)
        else:
            image = image_path_or_pil
        
        # Resize for display
        display_size = (400, 400)
        image = ImageOps.fit(image, display_size)
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None


# Routes
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/decode', methods=['POST'])
def api_decode():
    """API endpoint to decode uploaded image."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Decode image
        result = decode_image(filepath)
        
        # Convert image to base64 for display
        image_base64 = image_to_base64(filepath)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'result': result,
            'image': image_base64
        })
    
    except Exception as e:
        logger.error(f"Error in /api/decode: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/decode-base64', methods=['POST'])
def api_decode_base64():
    """API endpoint to decode base64 encoded image."""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Decode
        result = decode_image(image)
        image_base64 = image_to_base64(image)
        
        return jsonify({
            'result': result,
            'image': image_base64
        })
    
    except Exception as e:
        logger.error(f"Error in /api/decode-base64: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def api_status():
    """Check if decoder is initialized."""
    try:
        if decoder is None:
            initialize_decoder()
        
        return jsonify({
            'ready': True,
            'device': str(device),
            'model_path': DECODER_CONFIG['model_path']
        })
    except Exception as e:
        logger.error(f"Error in /api/status: {e}")
        return jsonify({
            'ready': False,
            'error': str(e)
        }), 500


@app.route('/api/watermark', methods=['POST'])
def api_watermark():
    """API endpoint to apply visible + invisible watermark to image using HARVIM + StegaStamp."""
    import subprocess
    import tempfile
    import shutil
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        secret = request.form.get('secret', '')  # Optional secret for invisible watermark
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        if len(secret) > 5:
            return jsonify({'error': 'Secret message too long (max 5 characters)'}), 400
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save input image
            input_filename = secure_filename(file.filename)
            input_path = os.path.join(temp_dir, 'input_' + input_filename)
            file.save(input_path)
            
            logger.info(f"Applying watermark to: {input_path}")
            if secret:
                logger.info(f"With invisible secret: {secret}")
            
            # Get original image as base64
            original_image_base64 = image_to_base64(input_path)
            
            # Run watermarking script (visible + optional invisible)
            cmd = [
                'python',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'apply_visible_watermark.py'),
                '--image', input_path,
                '--output', temp_dir,
                '--save-original'
            ]
            
            # Add secret if provided
            if secret:
                cmd.extend(['--secret', secret])
            
            logger.info(f"Running watermark command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,  # Increased timeout for HARVIM optimization + StegaStamp
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode != 0:
                logger.error(f"Watermark command failed: {result.stderr}")
                logger.error(f"Watermark stdout: {result.stdout}")
                return jsonify({
                    'error': f'Watermarking failed',
                    'details': result.stderr or result.stdout
                }), 500
            
            # Find the watermarked output
            watermarked_path = os.path.join(temp_dir, 'watermarked_output.png')
            
            if not os.path.exists(watermarked_path):
                logger.error(f"Watermarked output not found at {watermarked_path}")
                # Look for any PNG file in temp directory
                png_files = [f for f in os.listdir(temp_dir) if f.endswith('.png')]
                if png_files:
                    watermarked_path = os.path.join(temp_dir, png_files[0])
                    logger.info(f"Found output: {watermarked_path}")
                else:
                    return jsonify({'error': 'No output file generated'}), 500
            
            # Convert watermarked image to base64
            watermarked_image_base64 = image_to_base64(watermarked_path)
            
            logger.info("Watermarking successful")
            
            watermark_type = "visible + invisible" if secret else "visible only"
            return jsonify({
                'success': True,
                'original_image': original_image_base64,
                'watermarked_image': watermarked_image_base64,
                'watermark_type': watermark_type,
                'message': f'Image watermarked successfully ({watermark_type})'
            })
        
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
    
    except subprocess.TimeoutExpired:
        logger.error("Watermark command timed out (HARVIM optimization took too long)")
        return jsonify({'error': 'Watermarking timed out. HARVIM optimization is computationally intensive. Please try again.'}), 500
    
    except Exception as e:
        logger.error(f"Error in /api/watermark: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Live Image Decoder Web App')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the app on (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0 - accessible from other machines)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model', type=str, help='Path to custom decoder model')
    
    args = parser.parse_args()
    
    # Update config if custom model provided
    if args.model:
        DECODER_CONFIG['model_path'] = args.model
    
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize decoder on startup
    try:
        initialize_decoder()
    except Exception as e:
        logger.warning(f"Could not initialize decoder on startup: {e}")
    
    # Run Flask app
    print(f"\n{'='*60}")
    print(f"Live Image Decoder Web App")
    print(f"{'='*60}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Device: {device}")
    print(f"Debug mode: {args.debug}")
    print(f"{'='*60}\n")
    
    app.run(debug=args.debug, host=args.host, port=args.port)
