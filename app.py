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

IMAGE_SIZE = (128, 128)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

MODEL_PATH = 'oil_spill_rf_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
try:
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")

app = Flask(__name__)

def read_image(file_stream):
    try:
        with Image.open(file_stream) as img:
            img = img.convert("RGB")
            return np.array(img)
    except:
        file_stream.seek(0)
        file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def preprocess_uploaded_image(file_stream):
    img = read_image(file_stream)
    if img is None:
        return None, None, None, None, None

    img_resized = transform.resize(img, IMAGE_SIZE, anti_aliasing=True)
    img_gray = color.rgb2gray(img_resized)
    hog_features, hog_image = hog(img_gray, visualize=True, **HOG_PARAMS)

    hist_r, _ = np.histogram(img_resized[:, :, 0], bins=32, range=(0, 1), density=True)
    hist_g, _ = np.histogram(img_resized[:, :, 1], bins=32, range=(0, 1), density=True)
    hist_b, _ = np.histogram(img_resized[:, :, 2], bins=32, range=(0, 1), density=True)
    color_hist = np.concatenate([hist_r, hist_g, hist_b])

    features = np.concatenate([hog_features, color_hist])
    return img, img_resized, img_gray, hog_image, features

@app.route('/', methods=['GET'])
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Oil Spill Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                text-align: center;
                padding: 20px;
            }
            .container {
                max-width: 500px;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                margin: auto;
            }
            h1 {
                color: #333;
            }
            input[type="file"] {
                display: block;
                margin: 10px auto;
            }
            button {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 15px;
                cursor: pointer;
                font-size: 16px;
                border-radius: 5px;
            }
            button:hover {
                background-color: #218838;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Oil Spill Detection</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <br>
                <button type="submit">Upload and Detect</button>
            </form>
        </div>
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

    features_scaled = scaler.transform([features])
    prediction = rf_model.predict(features_scaled)[0]
    prediction_proba = rf_model.predict_proba(features_scaled)[0]

    result_text = "Oil Spill Detected" if prediction == 1 else "No Oil Spill Detected"
    result_color = "red" if prediction == 1 else "green"

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                text-align: center;
                padding: 20px;
            }}
            .container {{
                max-width: 600px;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                margin: auto;
            }}
            h1 {{
                color: {result_color};
            }}
            .images {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
            }}
            .images img {{
                max-width: 250px;
                margin: 10px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            .btn {{
                display: inline-block;
                margin-top: 20px;
                background-color: #007bff;
                color: white;
                padding: 10px 15px;
                text-decoration: none;
                border-radius: 5px;
            }}
            .btn:hover {{
                background-color: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{result_text}</h1>
            <p>Probability of Oil Spill: {prediction_proba[1]:.2f}</p>
            <div class="images">
                <img src="data:image/png;base64,{image_to_base64(original)}" alt="Original">
                <img src="data:image/png;base64,{image_to_base64(resized)}" alt="Resized">
                <img src="data:image/png;base64,{image_to_base64(gray)}" alt="Grayscale">
                <img src="data:image/png;base64,{image_to_base64(hog_img)}" alt="HOG Features">
            </div>
            <a href="/" class="btn">Try Another Image</a>
        </div>
    </body>
    </html>
    '''

def image_to_base64(image):
    buf = io.BytesIO()
    plt.imsave(buf, image, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
