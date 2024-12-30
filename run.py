import streamlit as st
import torch
from PIL import Image
import os
import torchvision.transforms as T
import matplotlib.pyplot as plt
import io

sam_full_loaded = torch.load("sam_full_model.pth")

st.title("SAM Model: Image Segmentation")
st.write("Upload an image, input a text prompt, and see the segmented version.")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    text_prompt = st.text_input("Enter a text prompt (e.g., 'tree', 'car', etc.)")

    if text_prompt:
        st.write("Running SAM Model...")
        results = sam_full_loaded.predict(image_pil, text_prompt, box_threshold=0.24, text_threshold=0.24)

        st.write("Segmentation Output:")

        fig, ax = plt.subplots(figsize=(10, 10))
        sam_full_loaded.show_anns(
            cmap="Greens",
            box_color=None,
            blend=True,
            ax=ax  
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png")  
        buf.seek(0)

        segmented_image = Image.open(buf)
        st.image(segmented_image, caption="Segmented Image", use_column_width=True)
        buf.close()
