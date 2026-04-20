#!/usr/bin/env python
"""
HTTPS launcher for the web app - enables camera access on mobile devices.
Generates a self-signed certificate and runs the Flask app over HTTPS.

Usage:
    python launch_https.py [--port 5443] [--host 0.0.0.0]
"""

import os
import sys
import argparse
import ssl
import subprocess
from pathlib import Path

def create_self_signed_cert(cert_dir="certs"):
    """Create a self-signed SSL certificate for local development."""
    os.makedirs(cert_dir, exist_ok=True)
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")
    
    # Check if certificates already exist
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"✓ Using existing certificates: {cert_file}, {key_file}")
        return cert_file, key_file
    
    # Create self-signed certificate
    try:
        print("Generating self-signed SSL certificate...")
        cmd = [
            'openssl', 'req', '-x509', '-newkey', 'rsa:2048',
            '-keyout', key_file,
            '-out', cert_file,
            '-days', '365',
            '-nodes',
            '-subj', '/CN=localhost'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Certificate created: {cert_file}")
        print(f"✓ Key created: {key_file}")
        return cert_file, key_file
    except FileNotFoundError:
        print("✗ Error: 'openssl' not found. Install OpenSSL or run Flask app with HTTP.")
        print("  On Linux/Mac: brew install openssl (macOS) or apt-get install openssl (Linux)")
        print("  On Windows: Use HTTP mode instead")
        return None, None
    except Exception as e:
        print(f"✗ Error creating certificate: {e}")
        return None, None


def launch_https_app(port=5443, host="0.0.0.0", cert_file=None, key_file=None):
    """Launch Flask app over HTTPS."""
    
    if not cert_file or not key_file:
        print("✗ Cannot launch HTTPS without valid certificates")
        print("Falling back to HTTP mode (camera won't work on mobile)")
        launch_http_app(port, host)
        return
    
    print(f"\n{'='*70}")
    print(f"🔒 HTTPS Web App - Camera Enabled")
    print(f"{'='*70}")
    print(f"Access from THIS machine: https://localhost:{port}")
    print(f"Access from OTHER machines: https://<YOUR_IP>:{port}")
    print(f"Example: https://192.168.1.100:{port}")
    print(f"{'='*70}\n")
    print(f"⚠️  NOTE: Self-signed certificate - accept the security warning in browser")
    print(f"    Click 'Advanced' → 'Proceed' to continue\n")
    
    # Import Flask app
    from app import app
    
    # Create SSL context
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_file, key_file)
    
    # Run with SSL
    try:
        app.run(
            host=host,
            port=port,
            ssl_context=ssl_context,
            debug=False,
            use_reloader=False
        )
    except Exception as e:
        print(f"✗ Error running HTTPS server: {e}")
        sys.exit(1)


def launch_http_app(port=5000, host="0.0.0.0"):
    """Launch Flask app over HTTP (camera won't work on mobile)."""
    print(f"\n{'='*70}")
    print(f"⚠️  HTTP Mode - Camera Access Limited")
    print(f"{'='*70}")
    print(f"Access: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    print(f"From other machines: http://<YOUR_IP>:{port}")
    print(f"{'='*70}\n")
    print(f"NOTE: Camera will NOT work on mobile devices in HTTP mode")
    print(f"      Use HTTPS launcher instead: python launch_https.py\n")
    
    from app import app
    app.run(host=host, port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(
        description='HTTPS launcher for camera-enabled web app',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_https.py                          # HTTPS on 0.0.0.0:5443
  python launch_https.py --port 8443              # HTTPS on custom port
  python launch_https.py --http                   # HTTP mode (no camera on mobile)
  python launch_https.py --http --port 5000       # HTTP on custom port
        """
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5443,
        help='Port to run on (default: 5443 for HTTPS, 5000 for HTTP)'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--http',
        action='store_true',
        help='Run in HTTP mode (camera access disabled on mobile)'
    )
    parser.add_argument(
        '--cert',
        help='Path to custom certificate file'
    )
    parser.add_argument(
        '--key',
        help='Path to custom key file'
    )
    
    args = parser.parse_args()
    
    if args.http:
        # HTTP mode
        launch_http_app(args.port, args.host)
    else:
        # HTTPS mode
        if args.cert and args.key:
            # Use custom certificates
            cert_file, key_file = args.cert, args.key
            if not (os.path.exists(cert_file) and os.path.exists(key_file)):
                print(f"✗ Certificate or key file not found")
                sys.exit(1)
        else:
            # Create self-signed certificates
            cert_file, key_file = create_self_signed_cert()
            if not cert_file or not key_file:
                print("\nFalling back to HTTP mode...")
                launch_http_app(args.port, args.host)
                return
        
        launch_https_app(args.port, args.host, cert_file, key_file)


if __name__ == "__main__":
    main()
