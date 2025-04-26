  Face Obscuring & Compression - Streamlit App

This is a Streamlit application that allows users to:
- Detect and obscure faces in images and videos (via blurring or pixelating).
- Compress the obscured media using Run-Length Encoding (RLE) or Huffman Encoding.
- View compression statistics (size before and after, compression ratio).
- Download the processed images/videos and compressed data files.


  Features

- Modes: 
    - Image
    - Video
- Face Obscuring Methods:
    - Blur
    - Pixelate
- Compression Methods:
    - Run-Length Encoding (RLE)
    - Huffman Encoding
- Download Options:
    - Compressed Image/Video
    - Compressed Data (.pkl)


  How to Run Locally

1. Clone the repository:
    bash
    git clone https://github.com/komalnalage/facial_obscuring.git
    cd facial_obscuring
    
2. Install the dependencies:
    bash
    pip install -r requirements.txt
    
3. Run the Streamlit app:
    bash
    streamlit run app.py
    
 Project Structure

├── app.py                      Main Streamlit app

├── face_utils.py               Utility functions for face detection & obscuring

├── compression_utils.py        Utility functions for compression (RLE, Huffman)

├── requirements.txt            Python dependencies

├── README.md                   Project documentation 

└── videos/                     Temporary folder for uploaded videos


