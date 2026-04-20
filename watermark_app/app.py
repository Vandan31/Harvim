import streamlit as st
import numpy as np
import cv2
import random
import os
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_harvim import run_harvim_on_image


def add_visible_watermark(image, text):
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    h, w, _ = image.shape
    (tw, th), _ = cv2.getTextSize(text, font, 1, 2)

    x = random.randint(0, max(0, w - tw))
    y = random.randint(th, h)

    color = tuple([random.randint(100, 255) for _ in range(3)])

    cv2.putText(overlay, text, (x, y), font, 1, color, 2)
    return cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

def ecc_encode(text, repeat=3):
    return "|".join([text]*repeat)


def run_stegastamp(image_path, secret):
    base_dir = os.path.dirname(os.path.dirname(__file__))

    model_path = os.path.join(base_dir, "StegaStamp-pytorch", "asset", "best.pth")
    output_dir = os.path.join(base_dir, "StegaStamp-pytorch", "outputs", "demo")

    os.makedirs(output_dir, exist_ok=True)

    command = [
        sys.executable,
        "-m", "stegastamp.encode_image",
        "--model", model_path,
        "--image", image_path,
        "--save_dir", output_dir,
        "--secret", secret,
        "--height", "400",
        "--width", "400",
        "--secret_size", "100"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=os.path.join(base_dir, "StegaStamp-pytorch")
    )

    if result.returncode != 0:
        raise Exception(result.stderr)

    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, base + "_hidden.png")


def run_decoder(image_path):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "StegaStamp-pytorch", "asset", "best.pth")

    candidates = []

    for i in range(7):
        img = cv2.imread(image_path)
        if img is None:
            continue

        scale = 400 + i*2
        img = cv2.resize(img, (scale, scale))
        img = cv2.resize(img, (400, 400))

        temp_path = os.path.join(base_dir, f"decode_{i}.png")
        cv2.imwrite(temp_path, img)

        command = [
            sys.executable,
            "-m", "stegastamp.decode_image",
            "--model", model_path,
            "--image", temp_path
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=os.path.join(base_dir, "StegaStamp-pytorch")
        )

        decoded = result.stdout.strip().replace("|", "").strip()

        if decoded and decoded != "decode failed":
            candidates.append(decoded)

    if not candidates:
        return "decode failed"

    # Character-level voting
    max_len = max(len(s) for s in candidates)
    final = ""

    for i in range(max_len):
        chars = []
        for s in candidates:
            if i < len(s):
                chars.append(s[i])

        if chars:
            final += max(set(chars), key=chars.count)

    return final


def jpeg_attack(img, quality):
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, 1)

def blur_attack(img, k):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)

def crop_attack(img, ratio):
    h, w, _ = img.shape
    nh, nw = int(h * ratio), int(w * ratio)
    y = (h - nh) // 2
    x = (w - nw) // 2
    cropped = img[y:y+nh, x:x+nw]
    return cv2.resize(cropped, (w, h))

def whatsapp_attack(img):
    img = cv2.resize(img, (320, 320))
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
    img = cv2.imdecode(enc, 1)
    img = cv2.resize(img, (400, 400))
    return img


def decode_success(image_path, original_secret):
    try:
        decoded = run_decoder(image_path)
        return decoded.strip() == original_secret.strip(), decoded
    except:
        return False, ""


def run_benchmark(image, secret, base_dir):
    results = []

    attacks = [
        ("JPEG_65", lambda img: jpeg_attack(img, 65)),
        ("JPEG_75", lambda img: jpeg_attack(img, 75)),
        ("Blur_5", lambda img: blur_attack(img, 5)),
        ("Blur_9", lambda img: blur_attack(img, 9)),
        ("Crop_70%", lambda img: crop_attack(img, 0.7)),
        ("WhatsApp", lambda img: whatsapp_attack(img)),
    ]

    success_count = 0

    for name, fn in attacks:
        attacked = fn(image)

        temp_path = os.path.join(base_dir, f"{name}.png")
        cv2.imwrite(temp_path, attacked)

        success, decoded = decode_success(temp_path, secret)

        if success:
            success_count += 1

        results.append({
            "Attack": name,
            "Decoded": decoded,
            "Success": success
        })

    score = (success_count / len(attacks)) * 100
    return results, round(score, 2)


st.set_page_config(page_title="Watermark App", layout="wide")
st.title("🛡️ Watermarking App")

tab1, tab2 = st.tabs(["🔐 Watermark", "🔍 Decode"])


with tab1:
    file = st.file_uploader("Upload Image")
    mode = st.selectbox("Mode", ["Visible", "Invisible", "Hybrid"])
    text = st.text_input("Secret")

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        st.image(img, channels="BGR")

        if st.button("Apply"):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            img = cv2.resize(img, (400,400))

            path = os.path.join(base_dir, "temp.png")
            cv2.imwrite(path, img)

            if mode == "Visible":
                result = add_visible_watermark(img, text)

            elif mode == "Invisible":
                result = cv2.imread(run_stegastamp(path, text))

            else:
                hidden = cv2.imread(run_stegastamp(path, text))
                harvim = cv2.imread(run_harvim_on_image(path, os.path.join(base_dir, "data")))
                result = cv2.addWeighted(hidden, 0.9, harvim, 0.1, 0)

            st.image(result, channels="BGR")

            _, buf = cv2.imencode(".png", result)
            st.download_button("Download", buf.tobytes(), "output.png")


with tab2:
    file = st.file_uploader("Upload Image", key="decode")

    if file:
        path = "decode.png"
        with open(path, "wb") as f:
            f.write(file.read())

        if st.button("Decode"):
            st.success(run_decoder(path))

