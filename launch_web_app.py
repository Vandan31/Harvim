#!/usr/bin/env python3
"""
Live Image Decoder - Web App Launcher
Handles setup and launches the Flask application
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    """Print application header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Live Image Decoder - Web Application Launcher            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")

def check_python():
    """Check Python version"""
    print(f"{Colors.YELLOW}[1/5]{Colors.ENDC} Checking Python installation...")
    required_version = (3, 8)
    if sys.version_info < required_version:
        print(f"{Colors.RED}✗ Python {required_version[0]}.{required_version[1]}+ required{Colors.ENDC}")
        sys.exit(1)
    print(f"{Colors.GREEN}✓ Python {sys.version_info.major}.{sys.version_info.minor} found{Colors.ENDC}")

def check_pytorch():
    """Check PyTorch installation"""
    print(f"{Colors.YELLOW}[2/5]{Colors.ENDC} Checking PyTorch installation...")
    try:
        import torch
        print(f"{Colors.GREEN}✓ PyTorch {torch.__version__} found{Colors.ENDC}")
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"  → Device: {device}")
        return True
    except ImportError:
        print(f"{Colors.YELLOW}⚠ PyTorch not installed${Colors.ENDC}")
        return False

def check_dependencies():
    """Check Flask and other dependencies"""
    print(f"{Colors.YELLOW}[3/5]{Colors.ENDC} Checking dependencies...")
    required = ['flask', 'flask_cors', 'bchlib', 'PIL']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"{Colors.YELLOW}⚠ Missing packages: {', '.join(missing)}{Colors.ENDC}")
        print(f"  Run: pip install -r web_requirements.txt")
        return False
    
    print(f"{Colors.GREEN}✓ All dependencies installed{Colors.ENDC}")
    return True

def check_model():
    """Check for model file"""
    print(f"{Colors.YELLOW}[4/5]{Colors.ENDC} Checking model files...")
    model_path = Path("StegaStamp-pytorch/asset/best.pth")
    
    if model_path.exists():
        size = model_path.stat().st_size / (1024 * 1024)
        print(f"{Colors.GREEN}✓ Model found ({size:.1f} MB){Colors.ENDC}")
        return True
    else:
        print(f"{Colors.YELLOW}⚠ Model not found at {model_path}{Colors.ENDC}")
        print(f"  The app will run but decoding may not work")
        return False

def create_directories():
    """Create required directories"""
    print(f"{Colors.YELLOW}[5/5]{Colors.ENDC} Creating required directories...")
    dirs = ['templates', 'static/css', 'static/js', 'output']
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    os.makedirs('/tmp/image_uploads', exist_ok=True)
    print(f"{Colors.GREEN}✓ Directories created{Colors.ENDC}")

def find_free_port(start_port=5000, max_attempts=10):
    """Find an available port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    
    return start_port

def main():
    parser = argparse.ArgumentParser(
        description='Launch Live Image Decoder Web App',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_web_app.py                    # Run on default port
  python launch_web_app.py --port 8080        # Run on custom port
  python launch_web_app.py --debug            # Enable debug mode
  python launch_web_app.py --model /path/to/model.pth  # Use custom model
        """
    )
    
    parser.add_argument('--port', type=int, default=None,
                       help='Port to run on (default: 5000 or first free port)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0 - accessible from other machines)')
    parser.add_argument('--model', type=str,
                       help='Path to custom decoder model')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open browser automatically')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency checks')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Run checks unless skipped
    if not args.skip_checks:
        check_python()
        check_pytorch()
        if not check_dependencies():
            print(f"{Colors.RED}✗ Please install dependencies: pip install -r web_requirements.txt{Colors.ENDC}")
            sys.exit(1)
        check_model()
        create_directories()
    
    # Find available port
    port = args.port or find_free_port(5000)
    
    print()
    print(f"{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.GREEN}Setup completed!{Colors.ENDC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print()
    print(f"🚀 Starting web app...")
    print(f"   URL: {Colors.CYAN}http://{args.host}:{port}{Colors.ENDC}")
    print(f"   Debug: {Colors.YELLOW}{'ON' if args.debug else 'OFF'}{Colors.ENDC}")
    print()
    print(f"Press {Colors.YELLOW}Ctrl+C{Colors.ENDC} to stop\n")
    
    # Try to open browser
    if not args.no_browser:
        try:
            import webbrowser
            webbrowser.open(f"http://{args.host}:{port}")
        except:
            pass
    
    # Build command
    cmd = [sys.executable, 'app.py', '--port', str(port), '--host', args.host]
    
    if args.debug:
        cmd.append('--debug')
    
    if args.model:
        cmd.extend(['--model', args.model])
    
    # Launch app
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.ENDC}")
        sys.exit(0)

if __name__ == '__main__':
    main()
