# KTP and NPWP OCR Application

This project is an Optical Character Recognition (OCR) system for processing Indonesian KTP (National Identity Cards) and NPWP (Taxpayer Identification Cards). It leverages a custom-built card classifier and the "stepfun/GOT_OCR2" model from Hugging Face for OCR.

## Features

- **Card Classification**: A custom-built binary classifier to identify whether the uploaded image is a KTP or NPWP.
- **KTP & NPWP OCR Pipelines**: Utilizes the "stepfun/GOT_OCR2" model from Hugging Face to extract relevant details from the classified cards.
- **Efficient Model Management**: The model is loaded once and shared between the KTP and NPWP OCR pipelines.
- **Error Handling**: Descriptive error messages are returned in case of failures during processing or classification.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/username/ktp-npwp-ocr.git
cd ktp-npwp-ocr
```

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Download the Hugging Face `GOT_OCR2` Model
Before running the application, you need to download the `GOT_OCR2` model from Hugging Face. Follow these steps:

1. Go to the [Hugging Face model page](https://huggingface.co/stepfun/GOT_OCR2).
2. Download the model files and save them to your local machine.
3. Update the model path in your `.env` file:
   ```
   MODEL_PATH=/path/to/downloaded/GOT_OCR2/model
   ```

### 4. Set Up the Environment
Create a `.env` file in the root directory with the following content:
```
GROQ_API_KEY=<YOUR API KEY>
```

### 5. Running the Application
To run the Flask application, execute:
```bash
python server.py
```

### 6. Model and Dataset
- **OCR Model**: The project uses the "stepfun/GOT_OCR2" model from Hugging Face for OCR. Ensure that it is properly integrated and accessible.
- **Card Classifier**: The classifier for KTP and NPWP cards has been built from scratch.
- **Dataset**: The dataset used for training the card classifier is **not** included in this repository.

## Usage

1. Upload an image of a KTP or NPWP card.
2. The system classifies the card and processes it using the appropriate OCR pipeline.
3. Results, including extracted data, will be returned in JSON format.

## Technologies Used

- **Flask**: For serving the web API.
- **PyTorch**: To manage the custom-built card classifier.
- **Hugging Face**: "stepfun/GOT_OCR2" model for OCR.
- **Python**: The core language used for the entire pipeline.
- **OpenCV**: For preprocessing the images before OCR.
```

This version includes clear instructions for users to download the `GOT_OCR2` model and specify the model path in their environment configuration. Let me know if you need any further modifications!
