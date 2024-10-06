# app.py
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import logging
from PIL import Image
from io import BytesIO
import json
from load_and_classify import ModelLoader, classify_card  # Import ModelLoader here
import numpy as np
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for sessions
cors = CORS(app, resources={r'/*': {'origins': '*'}})
app.logger.setLevel(logging.ERROR)

# Generate a new UUID for each new user session
@app.before_request
def create_user_session():
    if 'uuid' not in session:
        session['uuid'] = str(uuid.uuid4())  # Store the generated UUID in the session
        print(f"New user session created with UUID: {session['uuid']}")

@app.route('/classify_and_ocr', methods=['POST'])
def classify_and_ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    # image = Image.open(BytesIO(image_file.read()))
    image = Image.open(image_file)

    # Retrieve the classifier model from ModelLoader
    classifier_model = ModelLoader.get_classifier_model()

    # Use the classifier model to classify the card (KTP or NPWP)
    label = classify_card(image, classifier_model)
    print(f"Label: {label}")

    user_uuid = session['uuid']  # Retrieve the UUID from the session

    if label == 0:  # KTP
        try:
            # Get the KTPOCR instance from ModelLoader
            ktp_ocr = ModelLoader.get_ktp_ocr()            
            
            # Perform OCR using the KTPOCR instance
            ktp_ocr_result = ktp_ocr.ocr_image(user_uuid, image)
            ktp_clean_text = ktp_ocr.clean_text(ktp_ocr_result)
            ktp_json_result = ktp_ocr.extract_information(ktp_clean_text)

            print(ktp_ocr_result)
            print(ktp_clean_text)

            if isinstance(ktp_json_result, str):
                ktp_json_result = json.loads(ktp_json_result)

            return jsonify({
                'code': 200,
                'message': 'Success to read OCR',
                'errors': None,
                'data': ktp_json_result
            })
        except Exception as e:
            return jsonify({
                'code': 500,
                'message': 'OCR processing failed',
                'errors': str(e),
                'data': None
            })
        
    elif label == 1:  # NPWP
        try:
            # Get the NPWPOCR instance from ModelLoader
            npwp_ocr = ModelLoader.get_npwp_ocr()

            # Perform OCR using the NPWPOCR instance
            npwp_ocr_result = npwp_ocr.ocr_image(user_uuid, image)
            npwp_clean_text = npwp_ocr.clean_text(npwp_ocr_result)
            npwp_json_result = npwp_ocr.extract_information(npwp_clean_text)

            print(npwp_ocr_result)
            print(npwp_clean_text)

            if isinstance(npwp_json_result, str):
                npwp_json_result = json.loads(npwp_json_result)

            return jsonify({
                'code': 200,
                'message': 'Success to read OCR',
                'errors': None,
                'data': npwp_json_result
            })
        except Exception as e:
            return jsonify({
                'code': 500,
                'message': 'OCR processing failed',
                'errors': str(e),
                'data': None
            })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
