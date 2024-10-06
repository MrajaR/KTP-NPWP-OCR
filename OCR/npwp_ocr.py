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
from constant import npwp_patterns

class NPWPInformation():
  def __init__(self):
    self.npwpId = ''
    self.nama = ''
    self.nik = ''
    self.alamat = ''
    self.kpp = ''
    self.tanggal_terdaftar = ''

class NPWPOCR():
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer

        # Initialize the model with custom parameters
        self.model = model
        
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "you are an assistant that are very proficient at fixing typo in indonesian NPWP (Nomor Pokok Wajib Pajak) with this format structure:\nNPWP : <NOMOR NPWP>\n<NAMA>\nNIK : <NOMOK NIK>\n<ALAMAT>\nKPP <KANTOR PELAYANAN PAJAK>\nTerdaftar : <tanggal-bulan-tahun>\ndon't forget to fix the typo by considering context"
                ),
                ("human", "fix typo in this text so that it will match indonesian NPWP card format, without you saying any explanation nor any word:\n{input}"),
            ]
            
        )

        self.llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")
        self.fix_typo_chain = self.prompt_template | self.llm

        self.result = NPWPInformation()

    def ocr_image(self, uuid, image_file: str, ocr_type: str = 'format'):
        """
        Perform OCR prediction on an input image.

        :param image_file: Path to the image file.
        :param ocr_type: Type of OCR ('ocr', 'format', etc.).
        :return: OCR result.
        """
        image_path = self.preprocess_img(image_file, uuid)

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
        img = np.array(img)

        # Convert RGB to BGR (since OpenCV expects BGR format)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Get the shape of the image and crop it
        top, right, channels = img.shape
        right = int(0.15 * right)
        cropped_img = img[right:, :]

        # Convert the image to grayscale
        # gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        preprocessed_npwp_path = os.path.join('preprocessed_card_img', 'npwp')
        os.makedirs(preprocessed_npwp_path, exist_ok=True)

        preprocessed_img_path = os.path.join('preprocessed_card_img', 'npwp', f'{uuid}.jpg')
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
    

    def extract_information(self, raw_text):
        # Define the regex patterns in a dictionary

        for key, pattern in npwp_patterns.items():
            match = re.search(pattern, raw_text)
            if match:
                # Assign extracted value to the corresponding attribute
                setattr(self.result, key, match.group(1).strip())

        return self.convert_to_json()

    def convert_to_json(self):
        # Convert the result object to JSON format
        return json.dumps(self.result.__dict__, indent=4)
