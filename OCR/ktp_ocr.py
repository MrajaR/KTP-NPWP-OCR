import torch
from transformers import AutoModel, AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import json
import cv2

import os
import re
from constant import ktp_patterns

load_dotenv()

class KTPInformation():
    def __init__(self):
        self.nama = ''
        self.tempat_lahir = ''
        self.tgl_lahir = ''
        self.alamat = ''
        self.agama = ''
        self.pernikahan = ''
        self.pekerjaan = ''
        self.golongan_darah = ''
        self.jenis_kelamin = ''
        self.kecamatan = ''
        self.rtrw = ''
        self.nik = ''
        self.kel_desa = ''
        self.kewarganegaraan = ''
        self.provinsi = ''
        self.masa_berlaku = 'seumur hidup'


class KTPOCR:
    def __init__(self, model, tokenizer):
        """
        Initialize the model and tokenizer.

        :param model_path: Path to the saved model weights (FP16).
        :param tokenizer_path: Path or name of the tokenizer (default is 'ucaslcl/GOT-OCR2_0').
        :param fp16_model_path: Path to the FP16 weights file.
        """
        # Load the tokenizer with custom parameters
        self.tokenizer = tokenizer

        # Initialize the model with custom parameters
        self.model = model
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "you are an assistant that are very proficient at fixing typo in indonesian ID card (KTP) with this format structure:\nPROVINSI : <PROVINCE>\nKOTA/KABUPATEN : <CITY/DISTRICT>\nNIK : <NIK_NUMBER>\nNama : <FULL_NAME>\nTempat/Tgl Lahir : <BIRTH_PLACE>, <BIRTH_DATE>\nJenis Kelamin : <GENDER>\nGol. Darah : <BLOOD_TYPE>\nAlamat : <ADDRESS>\nRT/RW : <RT>/<RW>\nKel/Desa : <SUBDISTRICT/VILLAGE>\nKecamatan : <SUB-DISTRICT>\nAgama : <RELIGION>\nStatus Perkawinan : <MARITAL_STATUS>\nPekerjaan : <OCCUPATION>\nKewarganegaraan : <NATIONALITY>\nBerlaku Hingga : <VALID_UNTIL>"
                ),
                ("human", "fix typo in this text so that it will match indonesian ID card (KTP) format, if gol. darah is not specified then make it '-', without you saying any explanation nor any word:\n{input}"),
            ]
            
        )

        self.llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")
        self.fix_typo_chain = self.prompt_template | self.llm

        self.result = KTPInformation()


    def ocr_image(self, uuid, image_file: str, ocr_type: str = 'ocr'):
        """
        Perform OCR prediction on an input image.

        :param image_file: Path to the image file.
        :param ocr_type: Type of OCR ('ocr', 'format', etc.).
        :return: OCR result.
        """
        image_path = self.preprocess_img(uuid, image_file)

        return self.model.chat(self.tokenizer, image_path, ocr_type=ocr_type)

    def preprocess_img(self, img, uuid):
        """
        Preprocess an image for OCR.

        Parameters
        ----------
        img : str
            Path to the image file.
        uuid : str
            Unique identifier for the image.

        Returns
        -------
        preprocessed_img_path : str
            Path to the preprocessed image file.
        """
        img = Image.open(img)
        img = np.array(img)

        # Convert RGB to BGR (since OpenCV expects BGR format)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Get the shape of the image and crop it
        top, right, channels = img.shape
        right = int(right * 0.75)
        cropped_img = img[0:top, 0:right]

        # Convert the image to grayscale
        # gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        preprocessed_ktp_dir = os.path.join('preprocessed_card_img', 'ktp')
        os.makedirs(preprocessed_ktp_dir, exist_ok=True)

        preprocessed_img_path = os.path.join(preprocessed_ktp_dir, f'{uuid}.jpg')
        cv2.imwrite(preprocessed_img_path, cropped_img)
        print("berhasil preprocessing image")

        return preprocessed_img_path
    
    def clean_text(self, text):
        """
        Fix typo in the given text. The text is expected to be typo from Indonesian ID card (KTP).
        
        :param text: The text to fix typo from.
        :return: The text with typo fixed.
        """
        return self.fix_typo_chain.invoke({"input": text}).content
    
    def extract_information(self, fixed_text):
        """
        Extract information from the given text. The text is expected to be a fixed typo text from Indonesian ID card (KTP).
        
        :param fixed_text: The text to extract information from.
        """
        for key, pattern in ktp_patterns.items():
            match = re.search(pattern, fixed_text)
            if match:
                setattr(self.result, key, match.group(1).strip())

        return self.convert_to_json()

    def convert_to_json(self):
        """
        Convert the result to JSON format.

        :return: The result in JSON format.
        """
        return json.dumps(self.result.__dict__, indent=4)
        

