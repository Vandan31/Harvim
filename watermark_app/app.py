import streamlit as st
import numpy as np
import cv2
import os
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_harvim import run_harvim_on_image


# -------------------------------
# STEGASTAMP ENCODE (FIXED)
# -------------------------------
def run_stegastamp(image_path, secret):
    base_dir = os.path.dirname(os.path.dirname(__file__))

    model_path = os.path.join(base_dir, "StegaStamp-pytorch", "asset", "best.pth")
    output_dir = os.path.join(base_dir, "StegaStamp-pytorch", "outputs", "demo")

    os.makedirs(output_dir, exist_ok=True)

    # ✅ FIX: truncate to 7 chars (BCH limit)
    secret = (secret or "")[:7]

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
    output_path = os.path.join(output_dir, base + "_hidden.png")

    if not os.path.exists(output_path):
        raise Exception("StegaStamp failed to generate output")

    return output_path


# -------------------------------
# DECODER (UNCHANGED)
# -------------------------------
def run_decoder(image_path):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "StegaStamp-pytorch", "asset", "best.pth")

    candidates = []

    for i in range(7):
        img = cv2.imread(image_path)
        if img is None:
            continue

        scale = 400 + i * 2
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


# -------------------------------
# UI (UNCHANGED)
# -------------------------------
st.set_page_config(page_title="Watermark App", layout="wide")
st.title("🛡️ Watermarking App")

tab1, tab2 = st.tabs(["🔐 Watermark", "🔍 Decode"])


# -------------------------------
# WATERMARK TAB
# -------------------------------
with tab1:
    file = st.file_uploader("Upload Image")
    mode = st.selectbox("Mode", ["Visible", "Invisible", "Hybrid"])
    text = st.text_input("Secret")

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        if img is None:
            st.error("Invalid image")
        else:
            st.image(img, channels="BGR")

            if st.button("Apply"):
                try:
                    base_dir = os.path.dirname(os.path.dirname(__file__))
                    img = cv2.resize(img, (400, 400))

                    input_path = os.path.join(base_dir, "temp.png")
                    cv2.imwrite(input_path, img)

                    if mode == "Visible":
                        final_path = run_harvim_on_image(
                            input_path,
                            os.path.join(base_dir, "data")
                        )

                    elif mode == "Invisible":
                        final_path = run_stegastamp(input_path, text)

                    else:
                        hidden_path = run_stegastamp(input_path, text)

                        final_path = run_harvim_on_image(
                            hidden_path,
                            os.path.join(base_dir, "data")
                        )

                    result = cv2.imread(final_path)

                    if result is None:
                        st.error("Failed to load output image")
                    else:
                        st.image(result, channels="BGR")

                        _, buf = cv2.imencode(".png", result)
                        st.download_button("Download", buf.tobytes(), "output.png")

                except Exception as e:
                    st.error(f"Error: {str(e)}")


# -------------------------------
# DECODE TAB
# -------------------------------
with tab2:
    file = st.file_uploader("Upload Image", key="decode")

    if file:
        path = "decode.png"
        with open(path, "wb") as f:
            f.write(file.read())

        if st.button("Decode"):
            result = run_decoder(path)

            if result == "decode failed":
                st.error("❌ Decode failed")
            else:
                st.success(f"✔ {result}")