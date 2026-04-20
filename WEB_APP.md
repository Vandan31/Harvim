# 🎯 Live Image Decoder - Web Application

A complete web app for decoding images with confidence scoring and decoded text display.

---

## ⚡ Quick Start (1 Minute)

```bash
cd /home/hem436/Documents/harvim

# Install dependencies
pip install -r web_requirements.txt

# Run the app (accessible from other machines on network)
python3 launch_web_app.py
```

**Access from:**
- Same machine: `http://localhost:5000`
- Other machines: `http://<SERVER_IP>:5000` (e.g., `http://192.168.1.100:5000`)

---

## 🌐 Network Access (For Other Users)

### Finding Your Server IP Address

**Linux/Mac:**
```bash
hostname -I          # Shows all IP addresses
# or
ifconfig | grep "inet " # Find inet address (usually 192.168.x.x or 10.x.x.x)
```

**Windows:**
```cmd
ipconfig
# Look for "IPv4 Address" under your network adapter
```

### Access from Other Machines

Once you have the server IP (e.g., `192.168.1.100`):

1. On other machines, open browser to:
   ```
   http://192.168.1.100:5000
   ```

2. Or from command line:
   ```bash
   # Linux/Mac
   curl http://192.168.1.100:5000/api/status
   
   # Decode image
   curl -F "image=@photo.jpg" http://192.168.1.100:5000/api/decode
   ```

### Run on Specific Port

```bash
python3 app.py --port 8080 --host 0.0.0.0
```

### Run on Localhost Only (Secure)

```bash
python3 app.py --host 127.0.0.1
```

---

| Feature | Description |
|---------|-------------|
| 📤 **Image Upload** | Drag & drop or browse (PNG, JPG, GIF, BMP) |
| 📷 **Webcam** | Real-time camera capture |
| 📊 **Confidence** | See reliability (0-100%) of each decode |
| 💬 **Message Display** | Shows extracted hidden message |
| 📜 **History** | Timestamped log of all attempts |
| 📈 **Statistics** | Success rate, average confidence |
| 🔌 **REST API** | Integration endpoints |
| ⚡ **GPU Support** | Automatic CUDA acceleration |

---

## 📁 Project Structure

```
harvim/
├── app.py                    ← Backend Flask server
├── launch_web_app.py         ← Recommended launcher
├── run_web_app.sh           ← Bash launcher
├── web_requirements.txt      ← Dependencies
├── templates/
│   └── index.html           ← Web interface
└── static/
    ├── css/style.css        ← Styling
    └── js/script.js         ← Client logic
```

---

## 🚀 Installation

### Option 1: Python Launcher (Recommended)
```bash
python3 launch_web_app.py
```
Auto-checks dependencies and opens browser.

### Option 2: Bash Launcher
```bash
./run_web_app.sh
```

### Option 3: Manual
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r web_requirements.txt
python3 app.py --port 5000
```

---

## 💻 Usage

### Web Interface
1. Click **"Select Image"** or drag image onto the upload area
2. Wait for decoding to complete
3. View **Confidence Score** (0-100%)
4. See **Decoded Message** 
5. Check **Decode History** log

### Webcam Mode
1. Click **"Start Webcam"** (grant camera permissions)
2. Click **"Capture & Decode"**
3. View results instantly

---

## 🔌 REST API

### Check Status
```bash
GET /api/status
```
Response:
```json
{
  "ready": true,
  "device": "cuda"
}
```

### Decode Image (Upload)
```bash
POST /api/decode
Content-Type: multipart/form-data

image: <file>
```
Response:
```json
{
  "result": {
    "success": true,
    "message": "Hidden message",
    "confidence": 0.92,
    "error": null
  },
  "image": "data:image/png;base64,..."
}
```

### Decode Base64 Image
```bash
POST /api/decode-base64
Content-Type: application/json

{"image": "data:image/png;base64,..."}
```

### Python Example
```python
import requests

# Upload image
with open('photo.jpg', 'rb') as f:
    result = requests.post(
        'http://localhost:5000/api/decode',
        files={'image': f}
    ).json()

print(f"Message: {result['result']['message']}")
print(f"Confidence: {result['result']['confidence']*100:.1f}%")
```

### cURL Example
```bash
curl -F "image=@photo.jpg" http://localhost:5000/api/decode
```

---

## 📊 Understanding Confidence

| Score | Meaning |
|-------|---------|
| 80-100% | ✓ Very high, message is correct |
| 60-80% | ✓ Good confidence, likely correct |
| 40-60% | ⚠ Medium confidence, verify result |
| 0-40% | ✗ Low confidence, don't trust |

---

## ⚙️ Configuration

### Default Configuration (Network Access)
By default, the app runs on `0.0.0.0` (accessible from all network interfaces):

```bash
python3 launch_web_app.py
# Accessible as: http://localhost:5000 (local)
#                http://<YOUR_IP>:5000 (other machines)
```

### Change Port & Host
```bash
python3 app.py --port 8080 --host 0.0.0.0
```

### Localhost Only (Secure)
```bash
python3 app.py --host 127.0.0.1 --port 5000
# Only accessible from this machine
```

### Custom Model Path
```bash
python3 launch_web_app.py --model /path/to/model.pth
```

### Debug Mode
```bash
python3 launch_web_app.py --debug
```

### Model Configuration (in app.py)
```python
DECODER_CONFIG = {
    'secret_size': 100,
    'height': 400,
    'width': 400,
    'model_path': './StegaStamp-pytorch/asset/best.pth'
}
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Port already in use** | Use different port: `--port 8080` |
| **Webcam not working** | Check browser permissions, try HTTPS for remote |
| **Model not found** | Verify path exists: `ls -lh StegaStamp-pytorch/asset/best.pth` |
| **Low confidence** | Try higher quality images |
| **GPU not detected** | Check CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"` |
| **Dependencies missing** | Run: `pip install -r web_requirements.txt` |

### Verify Setup
```bash
python3 -c "
import torch, flask, PIL
print('✓ All dependencies installed')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ GPU: {torch.cuda.is_available()}')
"
```

---

## 📋 API Response Fields

### Result Object
```json
{
  "success": boolean,      // Decode succeeded
  "message": string|null,  // Extracted message
  "confidence": float,     // 0.0-1.0 reliability
  "error": string|null     // Error description
}
```

---

## 🖥️ System Requirements

- Python 3.8+
- 2GB RAM (4GB recommended)
- 500MB disk space
- Optional: CUDA 11.8+ for GPU (5-10x faster)

---

## 📦 Dependencies

```
Flask==2.3.0
flask-cors==4.0.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
bchlib==1.0.1
scikit-image>=0.20.0
numpy>=1.24.0
```

---

## 🎮 Deployment Options

### Development
```bash
python3 app.py --debug
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Production Security
- Add authentication in `app.py`
- Implement rate limiting
- Use HTTPS/TLS
- Set environment restrictions

---

## 🌟 Pro Tips

- **First load slower**: Model loading takes 2-5 seconds
- **Subsequent decodes**: 100-300ms on GPU, 500-1000ms on CPU
- **Better results**: Use high-quality, clear images
- **Batch processing**: Use API for multiple images
- **Statistics saved**: Browser LocalStorage keeps history
- **GPU speedup**: 5-10x faster than CPU

---

## 📁 File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Flask backend, decoder logic, API endpoints (~350 lines) |
| `templates/index.html` | Web interface (~200 lines) |
| `static/css/style.css` | Styling (~500 lines) |
| `static/js/script.js` | Client-side logic (~500 lines) |
| `launch_web_app.py` | Python launcher with checks |
| `run_web_app.sh` | Bash launcher |
| `web_requirements.txt` | Python dependencies |

---

## 🔐 Security Notes

### Current Implementation
- File upload validation (size, extension)
- Secure filename handling
- CORS enabled
- Error handling

### Production Additions
```python
# Add API key authentication
API_KEY = "your-secret-key"

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated
```

---

## 📞 Common Commands

```bash
# Start app
python3 launch_web_app.py

# Custom port
python3 app.py --port 8080

# Debug mode
python3 launch_web_app.py --debug

# Check status
curl http://localhost:5000/api/status

# Test decode
curl -F "image=@test.jpg" http://localhost:5000/api/decode

# Check GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# List all endpoints
grep "@app.route" app.py
```

---

## 🎯 Use Cases

1. **Testing Images** - Quickly verify image decoding
2. **Batch Processing** - Use API for multiple images
3. **Real-time Monitoring** - Webcam continuous capture
4. **Integration** - Embed decoder in other apps
5. **Educational** - Learn steganography concepts

---

## ✅ Verification

Check all components:
```bash
# Files
ls -lh app.py templates/index.html static/css/style.css static/js/script.js

# Python
python3 -c "import torch, flask, PIL; print('✓ Ready')"

# Dependencies
pip list | grep -E "Flask|torch|Pillow"
```

---

## 🚀 Next Steps

1. ✅ Run: `python3 launch_web_app.py`
2. 📸 Upload test image or use webcam
3. 📊 View confidence & message
4. 🔌 Test REST API for integration
5. 🎨 Customize UI if needed

---

**Status**: ✅ Ready to Use  
**Last Updated**: April 18, 2024

For issues: Check Flask console output and browser console (F12) for errors.
