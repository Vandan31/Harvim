"""Invisible watermark embedding and extraction using DCT steganography.

This module implements Quantization Index Modulation (QIM) in the frequency
domain to embed a hidden text message into an image imperceptibly. The Blue
channel of the image is divided into 8x8 blocks; one bit is embedded per block
by quantizing the [4, 1] DCT coefficient.

Typical usage:
    # Embed
    python invisible_watermark.py embed \
        --input image.png --message "Harvim" --output watermarked.png

    # Extract
    python invisible_watermark.py extract \
        --input watermarked.png --strength 25.0

References:
    - Chen & Wornell (2001): Quantization Index Modulation (QIM)
    - Wallace (1991): JPEG DCT overview
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Final

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BLOCK_SIZE: Final[int] = 8
"""Side length (pixels) of each DCT block. Matches the JPEG standard."""

EMBED_ROW: Final[int] = 4
EMBED_COL: Final[int] = 1
"""Mid-frequency DCT coefficient position used for bit embedding.
   Low-frequency (top-left) would shift visible brightness; high-frequency
   (bottom-right) is destroyed by compression. [4, 1] is a robust middle ground.
"""

BLUE_CHANNEL: Final[int] = 2
"""Index of the Blue channel in an RGB/RGBA image array.
   Human eyes are least sensitive to changes in blue, minimising visibility.
"""

LOSSLESS_EXTENSIONS: Final[tuple[str, ...]] = (".png", ".bmp", ".tiff", ".tif")
"""File extensions that guarantee lossless storage of DCT modifications."""

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
    """Convert a flat list of bits back to a UTF-8 string.

    Reads bits in groups of 8, assembles bytes, and stops at the first null
    byte (0x00) used as a message terminator. 
    """
    byte_list: list[int] = []
    null_found = False
    
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for b in bits[i : i + 8]:
            byte = (byte << 1) | b
            
        if byte == 0:
            null_found = True
            break  # null terminator — end of message
            
        byte_list.append(byte)

    message = bytes(byte_list).decode("utf-8", errors="replace")
    
    # Warn the user if no terminator was found (likely extracting noise)
    if not null_found and len(byte_list) > 0:
        logger.warning(
            "No null terminator found during extraction. The image may not "
            "be watermarked, or the watermark was destroyed by compression."
        )
        
    return message


# ──────────────────────────────────────────────────────────────────────────────
# Core watermark operations
# ──────────────────────────────────────────────────────────────────────────────

def _max_capacity(image_height: int, image_width: int) -> int:
    """Return the maximum number of bits embeddable in an image."""
    return (image_height // BLOCK_SIZE) * (image_width // BLOCK_SIZE)


def embed_watermark(
    image_path: str,
    message: str,
    output_path: str,
    strength: float = 25.0,
) -> None:
    """Embed an invisible watermark into an image using block-DCT QIM."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Warn early if the output format will destroy the watermark.
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in LOSSLESS_EXTENSIONS:
        logger.warning(
            "Output extension '%s' is lossy (e.g. JPEG). "
            "JPEG re-quantisation will destroy the embedded watermark. "
            "Use .png for reliable extraction.",
            ext,
        )

    # Open image and safely handle Alpha (Transparency) channels
    img = Image.open(image_path)
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")
        
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape[:2]

    # Encode message with a null-byte terminator
    bits = text_to_bits(message) + [0] * 8
    capacity = _max_capacity(h, w)

    if len(bits) > capacity:
        max_chars = (capacity - 8) // 8
        raise ValueError(
            f"Image ({w}x{h}) can hold at most {max_chars} characters, "
            f"but the message requires {len(bits)} bits for "
            f"{len(message)} characters. Use a shorter message or a larger image."
        )

    blue = img_array[:, :, BLUE_CHANNEL].copy()
    bit_idx = 0

    for row in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        if bit_idx >= len(bits):
            break
        for col in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            if bit_idx >= len(bits):
                break

            block = blue[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE]
            dct_block = dctn(block, norm="ortho")

            # QIM embedding
            coeff = dct_block[EMBED_ROW, EMBED_COL]
            bucket = np.floor(coeff / strength)
            if bits[bit_idx] == 1:
                dct_block[EMBED_ROW, EMBED_COL] = (bucket + 0.75) * strength
            else:
                dct_block[EMBED_ROW, EMBED_COL] = (bucket + 0.25) * strength

            blue[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE] = idctn(
                dct_block, norm="ortho"
            )
            bit_idx += 1

    # Clip values and update the blue channel
    img_array[:, :, BLUE_CHANNEL] = np.clip(blue, 0, 255)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # FIX: Use np.round to prevent truncation errors before casting to uint8
    Image.fromarray(np.round(img_array).astype(np.uint8)).save(output_path)

    logger.info("Watermark embedded -> %s", output_path)
    logger.info(
        "Message: '%s' | Bits embedded: %d / %d capacity | Strength: %.1f",
        message,
        bit_idx,
        capacity,
        strength,
    )


def extract_watermark(
    image_path: str,
    strength: float = 25.0,
) -> str:
    """Extract an invisible watermark from a previously watermarked image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Watermarked image not found: {image_path}")

    # Safely handle Alpha channels during extraction
    img = Image.open(image_path)
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")
        
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape[:2]
    blue = img_array[:, :, BLUE_CHANNEL]

    max_bits = _max_capacity(h, w)
    bits: list[int] = []

    for row in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        if len(bits) >= max_bits:
            break
        for col in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            if len(bits) >= max_bits:
                break

            block = blue[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE]
            dct_block = dctn(block, norm="ortho")
            coeff = dct_block[EMBED_ROW, EMBED_COL]

            # QIM decoding
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
        description="Embed or extract an invisible DCT watermark in an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── embed ──────────────────────────────────────────────────────────────
    embed_p = subparsers.add_parser(
        "embed",
        help="Embed an invisible text watermark into an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    embed_p.add_argument("--input",    required=True, help="Path to the source image.")
    embed_p.add_argument("--message",  required=True, help="Text message to hide.")
    embed_p.add_argument("--output",   required=True, help="Destination path (use .png).")
    embed_p.add_argument(
        "--strength", type=float, default=25.0,
        help="QIM step size. Higher = more robust, slightly more visible.",
    )

    # ── extract ────────────────────────────────────────────────────────────
    extract_p = subparsers.add_parser(
        "extract",
        help="Extract an invisible watermark from a watermarked image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    extract_p.add_argument("--input", required=True, help="Path to the watermarked image.")
    extract_p.add_argument(
        "--strength", type=float, default=25.0,
        help="Must match the strength used during embedding.",
    )

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
        )
    elif args.mode == "extract":
        extract_watermark(
            image_path=args.input,
            strength=args.strength,
        )


if __name__ == "__main__":
    main()