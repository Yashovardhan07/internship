from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Load the TensorFlow Lite models
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

models = {
    "potato": {
        "model_path": r"C:\Users\yasho\OneDrive\Desktop\internship\models\potato.tflite",
        "class_names": ["Early Blight", "Healthy", "Late Blight"],
    },
    "tomato": {
        "model_path": r"C:\Users\yasho\OneDrive\Desktop\internship\models\tomato.tflite",
        "class_names": ["Bacterial Spot", "Healthy", "Yellow Leaf Curl"],
    },
    "grapes": {
        "model_path": r"C:\Users\yasho\OneDrive\Desktop\internship\models\grape.tflite",
        "class_names": ["Powdery Mildew", "Healthy", "Downy Mildew"],
    },
    "apple": {
        "model_path": r"C:\Users\yasho\OneDrive\Desktop\internship\models\apple.tflite",
        "class_names": ["Apple Scab", "Healthy", "Apple Black Rot"],
    },
}

# Prepare models with interpreters
for key, value in models.items():
    value["interpreter"], value["input_details"], value["output_details"] = load_tflite_model(value["model_path"])

# Treatment dictionary
treatments = {
    "Early Blight": "Use fungicides like mancozeb or chlorothalonil and practice crop rotation.",
    "Late Blight": "Apply copper-based fungicides and remove infected plant debris.",
    "Bacterial Spot": "Use copper-based bactericides and avoid overhead watering.",
    "Yellow Leaf Curl": "Control whitefly population and plant resistant varieties.",
    "Apple Black Rot": "Prune infected areas and use fungicides like myclobutanil.",
    "Powdery Mildew": "Apply fungicides such as myclobutanil or potassium bicarbonate and improve air circulation around plants.",
    "Downy Mildew": "Apply fungicides such as chlorothalonil and ensure proper plant spacing.",
    "Apple Scab": "Use fungicides like captan or sulfur, and ensure good air circulation.",
    "Healthy": "No treatment required. Maintain good agricultural practices.",
}

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Default route: Model selection
@app.route('/')
def home():
    return render_template("model_selection.html")

# Upload page for a specific model
@app.route('/upload/<model_name>')
def upload(model_name):
    if model_name not in models:
        return "Model not found!", 404
    return render_template("upload.html", model_name=model_name.capitalize())

# Prediction route
@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if model_name not in models:
        return "Model not found!", 404

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process image
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict with the selected model
        interpreter = models[model_name]["interpreter"]
        input_details = models[model_name]["input_details"]
        output_details = models[model_name]["output_details"]
        class_names = models[model_name]["class_names"]

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        # Get output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Render result with treatment information
        return render_template(
            "result.html",
            model_name=model_name.capitalize(),
            predicted_class=predicted_class,
            confidence=round(confidence * 100, 2),
            treatment=treatments.get(predicted_class, "No treatment available."),
            image_filename=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
