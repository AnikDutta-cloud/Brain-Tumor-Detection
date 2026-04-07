
import os
import numpy as np
from flask import Flask, request, render_template_string, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input  # ✅ FIXED: Import EfficientNet preprocessing
import uuid

# ================= APP INITIALIZATION =================
app = Flask(__name__)

# ================= CONFIGURATION =================
MODEL_PATH = r'C:\Users\Anik\PycharmProjects\PythonProject3\model_efficientnet_b3.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# ✅ FIXED: 4-class names matching your training data order
# IMPORTANT: This order MUST match training_set.class_indices alphabetically
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODEL =================
print("🔄 Loading model...")
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# ================= HTML TEMPLATE (Same as your original) =================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Tumor Detection AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 600px;
            width: 100%;
            text-align: center;
        }
        header h1 { color: #333; font-size: 28px; margin-bottom: 10px; }
        header p { color: #666; font-size: 14px; margin-bottom: 30px; }
        .upload-section { margin: 30px 0; }
        .file-input-wrapper { position: relative; margin-bottom: 20px; }
        input[type="file"] { display: none; }
        .custom-file-upload {
            display: inline-block;
            padding: 15px 30px;
            background: #f0f0f0;
            border: 2px dashed #667eea;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            color: #667eea;
            transition: all 0.3s;
        }
        .custom-file-upload:hover { background: #667eea; color: white; }
        #fileName { display: block; margin-top: 10px; color: #666; font-size: 14px; }
        .predict-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 16px;
            border-radius: 50px;
            cursor: pointer;
            transition: transform 0.3s;
            font-weight: 600;
        }
        .predict-btn:hover { transform: scale(1.05); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4); }
        .alert { padding: 15px; border-radius: 10px; margin: 20px 0; font-size: 14px; }
        .error { background: #ffe6e6; color: #cc0000; border: 1px solid #cc0000; }
        .result-section { margin-top: 30px; padding-top: 30px; border-top: 2px solid #f0f0f0; }
        .result-section h2 { color: #333; margin-bottom: 20px; }
        .result-card { padding: 30px; border-radius: 15px; margin-bottom: 20px; animation: fadeIn 0.5s; }
        .result-card.tumor { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%); color: white; }
        .result-card.no-tumor { background: linear-gradient(135deg, #51cf66 0%, #40c057 100%); color: white; }
        .result-icon { font-size: 50px; margin-bottom: 10px; }
        .result-text { font-size: 24px; font-weight: 600; margin-bottom: 10px; }
        .confidence { font-size: 18px; opacity: 0.9; }
        .image-preview { margin: 20px 0; }
        .image-preview h4 { color: #333; margin-bottom: 10px; }
        .image-preview img { max-width: 100%; border-radius: 10px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); }
        .new-scan-btn {
            display: inline-block;
            padding: 12px 30px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 50px;
            margin-top: 20px;
            transition: background 0.3s;
        }
        .new-scan-btn:hover { background: #764ba2; }
        footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #f0f0f0; color: #999; font-size: 12px; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @media (max-width: 500px) {
            .container { padding: 20px; }
            header h1 { font-size: 22px; }
            .predict-btn { width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Brain Tumor Detection</h1>
            <p>Upload MRI Scan Image for AI-Powered Analysis</p>
        </header>
        <div class="upload-section">
            <form action="/predict" method="POST" enctype="multipart/form-data">
                <div class="file-input-wrapper">
                    <input type="file" name="file" id="fileInput" accept="image/*" required>
                    <label for="fileInput" class="custom-file-upload">📁 Choose MRI Image</label>
                    <span id="fileName">No file selected</span>
                </div>
                <button type="submit" class="predict-btn">🔍 Analyze Image</button>
            </form>
        </div>
        {% if error %}
        <div class="alert error">⚠️ {{ error }}</div>
        {% endif %}
        {% if prediction %}
        <div class="result-section">
            <h2>📊 Analysis Result</h2>
            <div class="result-card {{ 'tumor' if 'Tumor' in prediction else 'no-tumor' }}">
                <div class="result-icon">{% if 'Tumor' in prediction %}⚠️{% else %}✅{% endif %}</div>
                <h3 class="result-text">{{ prediction }}</h3>
                <p class="confidence">Confidence: {{ confidence }}%</p>
                {% if detailed_class %}
                <p style="font-size:14px; margin-top:5px; opacity:0.8;">Type: {{ detailed_class }}</p>
                {% endif %}
            </div>
            <div class="image-preview">
                <h4>Uploaded Scan:</h4>
                <img src="{{ img_path }}" alt="MRI Scan">
            </div>
            <a href="/" class="new-scan-btn">🔄 Analyze Another Image</a>
        </div>
        {% endif %}
        <footer>
            <p>⚕️ For Educational Purpose Only | Not a Medical Device</p>
        </footer>
    </div>
    <script>
        document.getElementById('fileInput').addEventListener('change', function(e) {
            var fileName = e.target.files[0] ? e.target.files[0].name : 'No file selected';
            document.getElementById('fileName').textContent = fileName;
        });
    </script>
</body>
</html>
"""


# ================= HELPER FUNCTIONS =================

# ✅ FIXED: Preprocessing function using EfficientNet's preprocess_input
def preprocess_image(img_path):
    """Preprocess image for EfficientNetB3 model"""
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ CRITICAL FIX: Use EfficientNet preprocessing (NOT simple /255 scaling)
    img_array = preprocess_input(img_array)

    return img_array


# ✅ FIXED: Prediction function for 4-class multi-class output
def predict_image(img_path):
    """Predict tumor type from MRI image"""
    processed_img = preprocess_image(img_path)

    # Get prediction (output shape: [1, 4] for 4 classes)
    prediction = model.predict(processed_img, verbose=0)[0]

    # ✅ FIXED: Find the class with highest probability (multi-class softmax)
    pred_index = np.argmax(prediction)
    confidence = float(prediction[pred_index])
    confidence_percent = round(confidence * 100, 2)

    # Get predicted class name
    predicted_class = CLASS_NAMES[pred_index]

    # Create display result (binary for UI)
    if predicted_class == 'notumor':
        display_result = "No Tumor Detected ✅"
    else:
        display_result = "Tumor Detected ⚠️"

    return display_result, confidence_percent, predicted_class


# ================= ROUTES =================

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template_string(HTML_TEMPLATE, error="❌ No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template_string(HTML_TEMPLATE, error="❌ No file selected!")

    # Validate file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return render_template_string(HTML_TEMPLATE,
                                      error="❌ Invalid file type. Please upload an image (PNG, JPG, JPEG).")

    # Save uploaded file with unique name
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
    img_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(img_path)

    try:
        # ✅ FIXED: Get prediction with detailed class info
        result, confidence, detailed_class = predict_image(img_path)

        # Create proper URL for displaying uploaded image
        relative_path = os.path.relpath(img_path, start='static').replace('\\', '/')
        img_url = url_for('static', filename=relative_path)

        return render_template_string(
            HTML_TEMPLATE,
            prediction=result,
            confidence=confidence,
            detailed_class=detailed_class.title(),  # Show tumor type: Glioma, Meningioma, etc.
            img_path=img_url
        )
    except Exception as e:
        if os.path.exists(img_path):
            os.remove(img_path)
        return render_template_string(HTML_TEMPLATE, error=f"❌ Prediction error: {str(e)}")


@app.errorhandler(404)
def not_found(error):
    return render_template_string(HTML_TEMPLATE, error="❌ Page not found!"), 404


@app.errorhandler(500)
def server_error(error):
    return render_template_string(HTML_TEMPLATE, error="❌ Server error. Please try again."), 500


# ================= RUN APP =================
if __name__ == '__main__':
    print("🚀 Starting Brain Tumor Detection Server...")
    print(f"📁 Model: {MODEL_PATH}")
    print(f"🖼️  Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"📂 Upload Folder: {UPLOAD_FOLDER}")
    print(f"🧠 Classes: {CLASS_NAMES}")
    print("🌐 Open in browser: http://127.0.0.1:5000/")
    print("-" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)