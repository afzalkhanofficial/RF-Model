import os
import io
import base64
import joblib
import cv2
import numpy as np
from flask import Flask, request, render_template_string, abort
from PIL import Image, ImageFile
from skimage import color, transform
from skimage.feature import hog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Allow truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants for image processing and feature extraction
IMAGE_SIZE = (128, 128)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# Load the trained Random Forest model and scaler
MODEL_PATH = 'oil_spill_rf_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
try:
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")

app = Flask(__name__)

def read_image(file_stream):
    """
    Attempts to read an image from a stream using PIL. Falls back to OpenCV if needed.
    Returns the image as a NumPy array in RGB format.
    """
    try:
        with Image.open(file_stream) as img:
            img = img.convert("RGB")
            return np.array(img)
    except Exception as e:
        try:
            file_stream.seek(0)
            file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading image: {e}")
            return None

def preprocess_uploaded_image(file_stream):
    """
    Processes the uploaded image:
    - Reads the image (supports all common formats)
    - Resizes it to a fixed size
    - Converts it to grayscale for HOG extraction
    - Extracts HOG features and computes a color histogram
    Returns:
      original image, resized image, grayscale image, hog image, and the combined feature vector.
    """
    img = read_image(file_stream)
    if img is None:
        return None, None, None, None, None

    # Resize image to standard size
    try:
        img_resized = transform.resize(img, IMAGE_SIZE, anti_aliasing=True)
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None, None, None, None, None

    # Convert to grayscale for HOG extraction and visualization
    try:
        img_gray = color.rgb2gray(img_resized)
    except Exception as e:
        print(f"Error converting image to grayscale: {e}")
        return None, None, None, None, None

    # Extract HOG features and visualization
    try:
        hog_features, hog_image = hog(img_gray, visualize=True, **HOG_PARAMS)
    except Exception as e:
        print(f"Error extracting HOG features: {e}")
        return None, None, None, None, None

    # Compute color histogram for each RGB channel (32 bins per channel)
    try:
        hist_r, _ = np.histogram(img_resized[:, :, 0], bins=32, range=(0, 1), density=True)
        hist_g, _ = np.histogram(img_resized[:, :, 1], bins=32, range=(0, 1), density=True)
        hist_b, _ = np.histogram(img_resized[:, :, 2], bins=32, range=(0, 1), density=True)
        color_hist = np.concatenate([hist_r, hist_g, hist_b])
    except Exception as e:
        print(f"Error computing color histogram: {e}")
        color_hist = np.zeros(96)
    
    # Combine HOG features and color histogram into a single feature vector
    features = np.concatenate([hog_features, color_hist])
    return img, img_resized, img_gray, hog_image, features

def create_visual_report(original, resized, gray, hog_img, prediction, proba):
    """
    Generates visual reports: original image, grayscale, HOG, heatmap, and a probability bar chart.
    Returns a dictionary of base64-encoded images.
    """
    visuals = {}

    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)
        return encoded

    def plot_image(img_array, cmap=None, title=""):
        fig = Figure(figsize=(3,3))
        ax = fig.subplots()
        ax.axis('off')
        if title:
            ax.set_title(title)
        if cmap:
            ax.imshow(img_array, cmap=cmap)
        else:
            ax.imshow(img_array)
        return fig

    # Original Image
    fig_orig = plot_image(original, title="Original Image")
    visuals['original'] = fig_to_base64(fig_orig)

    # Resized Image
    fig_resized = plot_image(resized, title="Resized Image")
    visuals['resized'] = fig_to_base64(fig_resized)

    # Grayscale Image
    fig_gray = plot_image(gray, cmap='gray', title="Grayscale")
    visuals['grayscale'] = fig_to_base64(fig_gray)

    # HOG Extraction Visualization
    fig_hog = plot_image(hog_img, cmap='gray', title="HOG Extraction")
    visuals['hog'] = fig_to_base64(fig_hog)

    # Heatmap (using grayscale as base)
    try:
        heatmap = cv2.applyColorMap(np.uint8(gray * 255), cv2.COLORMAP_JET)
    except Exception as e:
        heatmap = gray
    fig_heat = plot_image(heatmap, title="Heatmap")
    visuals['heatmap'] = fig_to_base64(fig_heat)

    # Prediction Probability Chart
    fig_chart = Figure(figsize=(4,3))
    ax_chart = fig_chart.subplots()
    classes = ['No Oil Spill', 'Oil Spill']
    ax_chart.bar(classes, proba, color=['green', 'red'])
    ax_chart.set_ylim([0, 1])
    ax_chart.set_ylabel("Probability")
    ax_chart.set_title("Prediction Probabilities")
    visuals['prob_chart'] = fig_to_base64(fig_chart)

    result = "Oil Spill Detected" if prediction == 1 else "No Oil Spill Detected"
    color = "red" if prediction == 1 else "green"
    visuals['prediction_text'] = f"<span style='color: {color}; font-weight: bold; font-size: 1.3rem;'>{result}</span>"
    return visuals

@app.route('/', methods=['GET'])
def index():
    return '''
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Random Forest @afzalkhan</title>
    <style>
      body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(to right, #4A90E2, #357ABD);
        margin: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
      }

      .container {
        background: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        text-align: center;
        width: 70%;
        max-width: 400px;
      }

      h1 {
        color: #333;
        font-size: 2rem;
      }

      #drop-area {
        border: 3px dashed #4A90E2;
        padding: 25px;
        cursor: pointer;
        border-radius: 8px;
        transition: background 0.3s ease-in-out;
      }

      #drop-area:hover {
        background: #f4f4f9;
      }

      #drop-area.pressed {
        background: #e0e0e0;
      }

      #preview {
        margin-top: 20px;
        margin-bottom: 20px;
      }

      #preview img {
        max-width: 100%;
        max-height: 300px;
        border-radius: 8px;
      }

      button {
        background: #4A90E2;
        color: white;
        font-size: 1rem;
        border: none;
        padding: 12px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: background 0.3s;
      }

      button:hover {
        background: #357ABD;
      }

      .browse-text {
        color: #007bff;
        font-weight: bold;
        cursor: pointer;
        text-decoration: underline;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>RF Oil Spill Detection</h1>
      <form id="upload-form" action="/predict" method="post" enctype="multipart/form-data">
        <div id="drop-area">
          <p>Drag & Drop an image or <span class="browse-text">Browse</span></p>
          <input type="file" id="file-input" name="image" accept="image/*" required hidden>
        </div>
        <div id="preview"></div>
        <button type="submit">Upload & Detect</button>
      </form>
    </div>
    <script>
      document.addEventListener('DOMContentLoaded', function() {
        const dropArea = document.getElementById('drop-area');
        const fileInput = document.getElementById('file-input');
        const preview = document.getElementById('preview');
        const browseText = document.querySelector('.browse-text');

        dropArea.addEventListener('dragover', (e) => {
          e.preventDefault();
          dropArea.classList.add('pressed');
        });

        dropArea.addEventListener('dragleave', () => {
          dropArea.classList.remove('pressed');
        });

        dropArea.addEventListener('drop', (e) => {
          e.preventDefault();
          dropArea.classList.remove('pressed');
          const files = e.dataTransfer.files;
          handleFiles(files);
        });

        dropArea.addEventListener('click', () => {
          fileInput.click();
        });

        browseText.addEventListener('click', (e) => {
          e.stopPropagation();
          fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
          handleFiles(e.target.files);
        });

        function handleFiles(files) {
          if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
              const reader = new FileReader();
              reader.onload = function(e) {
                preview.innerHTML = `<img src="${e.target.result}" alt="Image preview">`;
              };
              reader.readAsDataURL(file);
            } else {
              alert('Please upload a valid image file.');
            }
          }
        }
      });
    </script>
  </body>
</html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        abort(400, "No image file uploaded.")

    file = request.files['image']
    if file.filename == "":
        abort(400, "No selected file.")

    original, resized, gray, hog_img, features = preprocess_uploaded_image(file.stream)
    if features is None:
        abort(400, "Error processing image. Please upload a valid image file.")

    # Scale the combined feature vector using the loaded scaler
    features_scaled = scaler.transform([features])

    # Get prediction and probability estimates
    prediction = rf_model.predict(features_scaled)[0]
    prediction_proba = rf_model.predict_proba(features_scaled)[0]

    # Generate visual report
    visuals = create_visual_report(original, resized, gray, hog_img, prediction, prediction_proba)

    html_response = f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Random Forest @afzalkhan</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
      :root {{
        --primary-color: #4A90E2;
        --secondary-color: #357ABD;
        --accent-color: #FF6B6B;
        --text-color: #2d3436;
      }}

      * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }}

      body {{
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #4A90E2, #357ABD);
        color: var(--text-color);
        min-height: 100vh;
        padding: 2rem;
        display: flex;
        justify-content: center;
        align-items: center;
      }}

      .container {{
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        padding: 2.5rem;
        width: 95%;
        max-width: 1200px;
        margin: 2rem auto;
        animation: fadeIn 0.6s ease-out;
      }}

      h1 {{
        color: var(--primary-color);
        font-size: 2.8rem;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 600;
      }}

      h2 {{
        color: var(--secondary-color);
        font-size: 2rem;
        margin: 2rem 0 1.5rem;
        text-align: center;
      }}

      .prediction-text {{
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 4px solid var(--accent-color);
        font-size: 1.2rem;
        transition: transform 0.3s ease;
      }}

      .prediction-text:hover {{
        transform: translateY(-3px);
      }}

      .content-section {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
      }}

      .image-card {{
        position: relative;
        overflow: hidden;
        background: white;
        border-radius: 15px;
        padding: 1.2rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
      }}

      .image-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
      }}

      .image-card h3 {{
        color: var(--secondary-color);
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 500;
      }}

      .image-card img {{
        width: 100%;
        height: 300px;
        object-fit: contain;
        border-radius: 8px;
        border: 2px solid #f1f3f5;
        object-position: center;
        transition: transform 0.3s ease;
        padding: 10px;
      }}

      .btn {{
        display: inline-flex;
        align-items: center;
        padding: 0.8rem 2rem;
        background: var(--primary-color);
        color: white;
        text-decoration: none;
        border-radius: 8px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        margin: 2rem auto 0;
        gap: 0.5rem;
      }}

      .btn:hover {{
        background: var(--secondary-color);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
      }}

      @media (max-width: 768px) {{
        .container {{
          padding: 1.5rem;
          border-radius: 15px;
        }}

        h1 {{
          font-size: 2.2rem;
        }}

        h2 {{
          font-size: 1.6rem;
        }}

        .image-card img {{
          height: 250px;
        }}

        .btn {{
          width: 100%;
          justify-content: center;
        }}
      }}

      @keyframes fadeIn {{
        from {{
          opacity: 0;
          transform: translateY(20px);
        }}

        to {{
          opacity: 1;
          transform: translateY(0);
        }}
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>🌊 Random Forest Analysis Report</h1>
      <div class="prediction-text">
        <p>{visuals['prediction_text']}</p>
      </div>
      <h2>📊 Visual Analysis</h2>
      <div class="content-section">
        <div class="image-card">
          <h3>Original Image</h3><img src="data:image/png;base64,{visuals['original']}" alt="Original Image" />
        </div>
        <div class="image-card">
          <h3>Grayscale Analysis</h3><img src="data:image/png;base64,{visuals['grayscale']}" alt="Grayscale Image" />
        </div>
        <div class="image-card">
          <h3>HOG Features</h3><img src="data:image/png;base64,{visuals['hog']}" alt="HOG Extraction" />
        </div>
        <div class="image-card">
          <h3>Heatmap Visualization</h3><img src="data:image/png;base64,{visuals['heatmap']}" alt="Heatmap" />
        </div>
        <div class="image-card">
          <h3>Probability Distribution</h3><img src="data:image/png;base64,{visuals['prob_chart']}" alt="Prediction Probabilities" />
        </div>
      </div><a href="/" class="btn"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-arrow-left">
          <line x1="19" y1="12" x2="5" y2="12"></line>
          <polyline points="12 19 5 12 12 5"></polyline>
        </svg> Analyze Another Image </a>
    </div>
  </body>
</html>
    """
    return html_response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
