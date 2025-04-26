import pickle
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io

from face_utils import detect_and_obscure_faces_image
from compression_utils import rle_compress, huffman_compress, calculate_compression_ratio

st.set_page_config(layout="wide")
st.title(" Face Obscuring & Compression - Image and Video")

mode = st.radio("Select Mode", ["Image", "Video"])
compression_method = st.selectbox("Choose Compression Method", ["RLE", "Huffman"])
obscure_method = st.selectbox("Choose Obscuring Method", ["Blur", "Pixelate"])

def convert_cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def get_size_kb(data):
    if isinstance(data, np.ndarray):
        return data.nbytes / 1024
    elif isinstance(data, (bytes, bytearray)):
        return len(data) / 1024
    elif isinstance(data, list):
        return len(data) * 2 / 1024
    elif isinstance(data, dict) and "compressed_data" in data:
        return (len(data["compressed_data"]) + len(str(data["codes"]).encode())) / 1024
    else:
        return len(pickle.dumps(data)) / 1024

# -------------------- IMAGE MODE --------------------
if mode == "Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        original = image.copy()

        obscured = detect_and_obscure_faces_image(image.copy(), method=obscure_method)

        if compression_method == "RLE":
            compressed = rle_compress(obscured)
        elif compression_method == "Huffman":
            compressed = huffman_compress(obscured)

        original_size_kb = get_size_kb(obscured)
        compressed_size_kb = get_size_kb(compressed)
        ratio = calculate_compression_ratio(obscured, compressed)

        col1, col2 = st.columns(2)
        with col1:
            st.image(convert_cv2_to_pil(original), caption="Original Image", use_container_width=True)
        with col2:
            st.image(convert_cv2_to_pil(obscured), caption="Obscured Image", use_container_width=True)

        st.markdown(f"""
        ### 📊 Compression Details:
        - 🖼️ Original Size: **{original_size_kb:.2f} KB**
        - 📦 Compressed Size: **{compressed_size_kb:.2f} KB**
        - 📉 Compression Ratio: **{ratio:.2f}%**
        """)

        st.download_button(
            label="📥 Download Compressed Image Data",
            data=pickle.dumps(compressed),
            file_name=f"compressed_image_{compression_method.lower()}.pkl",
            mime="application/octet-stream"
        )

# -------------------- VIDEO MODE --------------------
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video with faces", type=["mp4"])
    if uploaded_video:
        os.makedirs("videos", exist_ok=True)
        video_path = os.path.join("videos", uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        output_path = os.path.join("videos", f"processed_{uploaded_video.name}")
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_width, target_height = 640, 480

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        st.info("Processing video... please wait ⏳")
        progress_bar = st.progress(0)

        total_original_size_kb = 0
        total_compressed_size_kb = 0
        processed_count = 0
        frame_count = 0
        compressed_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (target_width, target_height))
            obscured = detect_and_obscure_faces_image(frame.copy(), method=obscure_method)

            if compression_method == "RLE":
                compressed = rle_compress(obscured)
            elif compression_method == "Huffman":
                compressed = huffman_compress(obscured)

            compressed_frames.append(compressed)

            total_original_size_kb += get_size_kb(obscured)
            total_compressed_size_kb += get_size_kb(compressed)

            out.write(obscured)
            processed_count += 1
            frame_count += 1

            if frame_count % 5 == 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()

        compression_ratio = (total_compressed_size_kb / total_original_size_kb) * 100 if total_original_size_kb else 0

        st.success(f"🎉 Done! Processed {processed_count} frames.")
        st.markdown(f"""
        ### 📊 Compression Summary for Video:
        - 🖼️ Total Original Size: **{total_original_size_kb:.2f} KB**
        - 📦 Total Compressed Size: **{total_compressed_size_kb:.2f} KB**
        - 📉 Overall Compression Ratio: **{compression_ratio:.2f}%**
        """)

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            label="📥 Download Processed Video",
            data=video_bytes,
            file_name=f"processed_{uploaded_video.name}",
            mime="video/mp4"
        )

        # Download button for compressed data as .pkl
        st.download_button(
            label="📦 Download Compressed Video Data",
            data=pickle.dumps(compressed_frames),
            file_name=f"compressed_video_{compression_method.lower()}.pkl",
            mime="application/octet-stream"
        )
