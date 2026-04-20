#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Live Image Decoder - Web App Setup & Launch              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python installation
echo -e "${YELLOW}[1/5]${NC} Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"

# Check/Create virtual environment
echo ""
echo -e "${YELLOW}[2/5]${NC} Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo ""
echo -e "${YELLOW}[3/5]${NC} Installing dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
if pip install -r web_requirements.txt; then
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Check model file
echo ""
echo -e "${YELLOW}[4/5]${NC} Checking model files..."
if [ -f "StegaStamp-pytorch/asset/best.pth" ]; then
    MODEL_SIZE=$(du -h "StegaStamp-pytorch/asset/best.pth" | cut -f1)
    echo -e "${GREEN}✓ Model found (${MODEL_SIZE})${NC}"
else
    echo -e "${YELLOW}⚠ Model file not found at StegaStamp-pytorch/asset/best.pth${NC}"
    echo -e "${YELLOW}  The app will still run but decoding may not work${NC}"
fi

# Create required directories
echo ""
echo -e "${YELLOW}[5/5]${NC} Creating required directories..."
mkdir -p templates static/css static/js output /tmp/image_uploads
echo -e "${GREEN}✓ Directories created${NC}"

# Check if port is in use
PORT=5000
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠ Port ${PORT} is already in use${NC}"
    read -p "Enter alternative port (default 5001): " ALT_PORT
    PORT=${ALT_PORT:-5001}
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Setup completed successfully!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Starting web app on ${BLUE}http://localhost:${PORT}${NC}"
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop the server"
echo ""
echo "Opening in browser..."
sleep 2

# Try to open in default browser
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:${PORT}" 2>/dev/null &
elif command -v open &> /dev/null; then
    open "http://localhost:${PORT}" 2>/dev/null &
fi

# Start the Flask app
python app.py --port $PORT
