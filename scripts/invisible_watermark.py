"""Invisible watermark embedding and extraction using DCT steganography.

This module implements Quantization Index Modulation (QIM) in the frequency
domain to embed a hidden text message into an image imperceptibly. 

Upgraded with Key-Based Scrambling: 
The sequence of 8x8 blocks is shuffled using a secret integer key, scattering 
the payload across the image for enhanced security and spatial robustness.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from typing import Final

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BLOCK_SIZE: Final[int] = 8
EMBED_ROW: Final[int] = 4
EMBED_COL: Final[int] = 1
BLUE_CHANNEL: Final[int] = 2
LOSSLESS_EXTENSIONS: Final[tuple[str, ...]] = (".png", ".bmp", ".tiff", ".tif")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Bit-string helpers
# ──────────────────────────────────────────────────────────────────────────────

def text_to_bits(text: str) -> list[int]:
    """Convert a UTF-8 string to a flat list of bits (MSB first)."""
    bits: list[int] = []
    for byte in text.encode("utf-8"):
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def bits_to_text(bits: list[int]) -> str:
    """Convert a flat list of bits back to a UTF-8 string."""
    byte_list: list[int] = []
    null_found = False
    
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for b in bits[i : i + 8]:
            byte = (byte << 1) | b
            
        if byte == 0:
            null_found = True
            break  
            
        byte_list.append(byte)

    message = bytes(byte_list).decode("utf-8", errors="replace")
    
    if not null_found and len(byte_list) > 0:
        logger.warning(
            "No null terminator found. Is the key/strength correct? "
            "Was the image watermarked or compressed?"
        )
        
    return message


# ──────────────────────────────────────────────────────────────────────────────
# Core watermark operations
# ──────────────────────────────────────────────────────────────────────────────

def _get_shuffled_coords(h: int, w: int, key: int) -> list[tuple[int, int]]:
    """Generate a pseudo-randomly shuffled list of 8x8 block coordinates."""
    coords = []
    for row in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        for col in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            coords.append((row, col))
            
    # Seed the RNG and shuffle deterministically
    random.seed(key)
    random.shuffle(coords)
    return coords


def _max_capacity(h: int, w: int) -> int:
    """Return the maximum number of bits embeddable in an image."""
    return (h // BLOCK_SIZE) * (w // BLOCK_SIZE)


def embed_watermark(
    image_path: str,
    message: str,
    output_path: str,
    strength: float = 25.0,
    key: int = 42,
) -> None:
    """Embed an invisible watermark using scrambled block-DCT QIM."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    ext = os.path.splitext(output_path)[1].lower()
    if ext not in LOSSLESS_EXTENSIONS:
        logger.warning("Output extension '%s' is lossy. Use .png.", ext)

    img = Image.open(image_path)
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")
        
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape[:2]

    bits = text_to_bits(message) + [0] * 8
    capacity = _max_capacity(h, w)

    if len(bits) > capacity:
        max_chars = (capacity - 8) // 8
        raise ValueError(f"Image holds max {max_chars} chars. Message too long.")

    blue = img_array[:, :, BLUE_CHANNEL].copy()
    
    # Get the randomized sequence of blocks
    shuffled_coords = _get_shuffled_coords(h, w, key)
    
    bit_idx = 0
    for row, col in shuffled_coords:
        if bit_idx >= len(bits):
            break

        block = blue[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE]
        dct_block = dctn(block, norm="ortho")

        coeff = dct_block[EMBED_ROW, EMBED_COL]
        bucket = np.floor(coeff / strength)
        if bits[bit_idx] == 1:
            dct_block[EMBED_ROW, EMBED_COL] = (bucket + 0.75) * strength
        else:
            dct_block[EMBED_ROW, EMBED_COL] = (bucket + 0.25) * strength

        blue[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE] = idctn(dct_block, norm="ortho")
        bit_idx += 1

    img_array[:, :, BLUE_CHANNEL] = np.clip(blue, 0, 255)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(np.round(img_array).astype(np.uint8)).save(output_path)

    logger.info("Watermark embedded -> %s", output_path)
    logger.info("Message: '%s' | Bits: %d/%d | Strength: %.1f | Key: %d", 
                message, bit_idx, capacity, strength, key)


def extract_watermark(
    image_path: str,
    strength: float = 25.0,
    key: int = 42,
) -> str:
    """Extract an invisible watermark using the secret key."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Watermarked image not found: {image_path}")

    img = Image.open(image_path)
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")
        
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape[:2]
    blue = img_array[:, :, BLUE_CHANNEL]

    max_bits = _max_capacity(h, w)
    bits: list[int] = []
    
    # Generate the EXACT same randomized sequence of blocks
    shuffled_coords = _get_shuffled_coords(h, w, key)

    for row, col in shuffled_coords:
        if len(bits) >= max_bits:
            break

        block = blue[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE]
        dct_block = dctn(block, norm="ortho")
        coeff = dct_block[EMBED_ROW, EMBED_COL]

        remainder = (coeff / strength) % 1.0
        bits.append(1 if remainder > 0.5 else 0)

    message = bits_to_text(bits)
    logger.info("Extracted message: '%s'", message)
    return message


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invisible_watermark",
        description="Embed or extract an invisible DCT watermark with Key Scrambling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── embed ──────────────────────────────────────────────────────────────
    embed_p = subparsers.add_parser("embed", help="Embed an invisible text watermark.")
    embed_p.add_argument("--input",    required=True, help="Path to the source image.")
    embed_p.add_argument("--message",  required=True, help="Text message to hide.")
    embed_p.add_argument("--output",   required=True, help="Destination path (use .png).")
    embed_p.add_argument("--strength", type=float, default=25.0, help="QIM step size.")
    embed_p.add_argument("--key",      type=int,   default=42,   help="Secret integer key for scattering.")

    # ── extract ────────────────────────────────────────────────────────────
    extract_p = subparsers.add_parser("extract", help="Extract an invisible watermark.")
    extract_p.add_argument("--input",    required=True, help="Path to the watermarked image.")
    extract_p.add_argument("--strength", type=float, default=25.0, help="Must match embed strength.")
    extract_p.add_argument("--key",      type=int,   default=42,   help="Must match embed secret key.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "embed":
        embed_watermark(
            image_path=args.input,
            message=args.message,
            output_path=args.output,
            strength=args.strength,
            key=args.key,
        )
    elif args.mode == "extract":
        extract_watermark(
            image_path=args.input,
            strength=args.strength,
            key=args.key,
        )

if __name__ == "__main__":
    main()