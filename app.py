from flask import Flask, render_template, request
import os
from flask import jsonify
from werkzeug.utils import secure_filename
import tensorflow
import numpy as np
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness

app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness

# Utility to strip unsupported args
def filter_kwargs(allowed_keys, kwargs):
    return {k: v for k, v in kwargs.items() if k in allowed_keys}

# Custom deserializers
def custom_random_flip(**kwargs):
    return RandomFlip(**filter_kwargs(['mode', 'seed'], kwargs))

def custom_random_rotation(**kwargs):
    return RandomRotation(**filter_kwargs(['factor', 'fill_mode', 'interpolation', 'seed'], kwargs))

def custom_random_zoom(**kwargs):
    return RandomZoom(**filter_kwargs(['height_factor', 'width_factor', 'fill_mode', 'interpolation', 'seed'], kwargs))

def custom_random_contrast(**kwargs):
    return RandomContrast(**filter_kwargs(['factor', 'seed'], kwargs))

def custom_random_brightness(**kwargs):
    return RandomBrightness(**filter_kwargs(['factor', 'value_range', 'seed'], kwargs))

# Registering the stripped versions
custom_objects = {
    "RandomFlip": custom_random_flip,
    "RandomRotation": custom_random_rotation,
    "RandomZoom": custom_random_zoom,
    "RandomContrast": custom_random_contrast,
    "RandomBrightness": custom_random_brightness
}


# Load the trained models
MODEL_PATH_1 = 'models/skin_disease_model.h5'
MODEL_PATH_2 = 'models/vgg16_skin_disease_model.h5'
MODEL_PATH_3 = 'models/model.h5'
MODEL_PATH_4 = 'models/trained_model.keras'

model_1 = load_model(MODEL_PATH_1)
model_2 = load_model(MODEL_PATH_2)
model_3 = load_model(MODEL_PATH_3)
#model_4 = load_model(MODEL_PATH_4, custom_objects=custom_objects)
#model_4 = keras.models.load_model(MODEL_PATH_4, custom_objects=custom_objects)


# Update class names based on folder structure used in training
CLASS_NAMES_1 = ['Acne', 'Atopic', 'Basal cell carcinoma']  
CLASS_NAMES_2 = ['Actinic Keratosis','Atopic Dermatisis','Benign Keratosis','Dermatofibroma','Melanocytic nevus','Melanoma','Squamous cell carcinoma','Tinea Ringworm Candidiasis','Vascular lesion']
CLASS_NAMES_3 = ['Actinic Keratosis','Atopic Dermatisis','Benign Keratosis','Dermatofibroma','Melanocytic nevus','Melanoma','Squamous cell carcinoma','Tinea Ringworm Candidiasis','Vascular lesion']
CLASS_NAMES_4 = ['Actinic Keratosis','Atopic Dermatisis','Benign Keratosis','Dermatofibroma','Melanocytic nevus']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict1', methods=['POST'])
def predict1():
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']
    if file.filename == '':
        return 'No file selected.'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model_1.predict(img_array)
    predicted_class = CLASS_NAMES_1[np.argmax(predictions[0])]
    confidence = round(np.max(predictions[0]) * 100, 2)
    return jsonify({'prediction': predicted_class})

@app.route('/predict2', methods=['POST'])
def predict2():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model_2.predict(img_array)
    predicted_class = CLASS_NAMES_2[np.argmax(predictions[0])]
    confidence = round(np.max(predictions[0]) * 100, 2)

    return jsonify({
        'prediction': predicted_class,
        'confidence': f"{confidence}%",
        'filename': filename
    })

@app.route('/predict3', methods=['POST'])
def predict3():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model_3.predict(img_array)
    predicted_class = CLASS_NAMES_3[np.argmax(predictions[0])]
    confidence = round(np.max(predictions[0]) * 100, 2)

    return jsonify({
        'prediction': predicted_class,
        'confidence': f"{confidence}%",
        'filename': filename
    })



# model_4 = tensorflow.keras.models.load_model('models/trained_model.keras')

@app.route('/predict4', methods=['POST'])
def predict4():
    try:
        # Get the data from the POST request
        data = request.get_json()

        # Assume the input data is a list of features to be used for prediction
        input_data = np.array(data['input']).reshape(1, -1)  # Reshape based on your model's input shape

        # Make a prediction
        prediction = model_4.predict(input_data)

        # Prepare the response (example assumes it's a classification problem)
        response = {
            'prediction': prediction.tolist()  # Convert numpy array to list for JSON serialization
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
