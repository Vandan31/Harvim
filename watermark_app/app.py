import streamlit as st
import cv2
import numpy as np

from watermark.visible import add_visible_watermark
from watermark.invisible import embed_watermark
from watermark.hybrid import hybrid_watermark

st.title("Watermark App")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
text = st.text_input("Enter Text")

mode = st.selectbox("Mode", ["Visible","Invisible","Hybrid"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR")

    if st.button("Apply"):
        if mode == "Visible":
            result = add_visible_watermark(image, text)
        elif mode == "Invisible":
            result = embed_watermark(image, text)
        else:
            result = hybrid_watermark(image, text)

        st.image(result, channels="BGR")