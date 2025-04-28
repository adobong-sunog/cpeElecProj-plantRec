import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('plant_disease_model.h5')

# Configuration
class_labels = ['Healthy', 'Powdery', 'Rust']
UPLOAD_FOLDER = 'static/uploads'
MAX_HISTORY = 5

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory history of last MAX_HISTORY predictions
prediction_history = []

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def manage_upload_history():
    # If more than MAX_HISTORY, delete oldest image file & record
    while len(prediction_history) > MAX_HISTORY:
        oldest = prediction_history.pop(0)
        try:
            os.remove(oldest['img_path'])
        except FileNotFoundError:
            pass

@app.route('/')
def index():
    # Show most recent first
    return render_template('index.html', history=prediction_history[::-1])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(request.url)

    file = request.files['file']
    # give each upload a unique name
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # run prediction
    img_tensor = preprocess_image(save_path)
    preds = model.predict(img_tensor)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx] * 100)
    prediction = class_labels[idx]

    # record in history
    prediction_history.append({
        'img_path': save_path,
        'prediction': prediction,
        'confidence': f"{confidence:.2f}"
    })
    manage_upload_history()

    return render_template('result.html',
                           prediction=prediction,
                           confidence=f"{confidence:.2f}",
                           img_path=save_path,
                           history=prediction_history[::-1])

@app.route('/clear_history', methods=['POST'])
def clear_history():
    for record in prediction_history:
        try:
            os.remove(record['img_path'])
        except FileNotFoundError:
            pass
    prediction_history.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
