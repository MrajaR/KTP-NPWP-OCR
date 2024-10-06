from torchvision import models, transforms
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from OCR import KTPOCR 
from OCR import NPWPOCR 
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelLoader:
    _instance = None
    _classifier_model = None
    _ktp_ocr = None
    _npwp_ocr = None

    @classmethod
    def get_instance(cls):
        """
        Get the shared instance of the model and tokenizer. If the instance does not exist, it is created by calling _load_model().
        """
        if cls._instance is None:
            cls._instance = cls._load_model()
        return cls._instance

    @classmethod
    def _load_model(cls):
        """
        Load the shared model (for KTPOCR and NPWPOCR) from local.
        Initialize KTPOCR and NPWPOCR instances here.
        """
        model_path = 'local_fp16_model_GOT_OCR'
        
        # Load the model and tokenizer localally (not downloading from huggingface)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, 
                                          trust_remote_code=True, 
                                          low_cpu_mem_usage=True, 
                                          device_map='cuda', 
                                          use_safetensors=True, 
                                          pad_token_id=tokenizer.eos_token_id,
                                          torch_dtype=torch.float16)

        # Initialize KTPOCR and NPWPOCR with the shared model and tokenizer
        cls._ktp_ocr = KTPOCR(model=model, tokenizer=tokenizer)
        cls._npwp_ocr = NPWPOCR(model=model, tokenizer=tokenizer)
        
        return model, tokenizer

    @classmethod
    def get_ktp_ocr(cls):
        """
        Get the shared instance of the KTPOCR model.

        If the instance does not exist, it is created by calling `get_instance()`.
        """
        if cls._ktp_ocr is None:
            cls.get_instance()
        return cls._ktp_ocr

    @classmethod
    def get_npwp_ocr(cls):
        """
        Get the shared instance of the NPWPOCR model.

        If the instance does not exist, it is created by calling `get_instance()`.
        """
        if cls._npwp_ocr is None:
            cls.get_instance()
        return cls._npwp_ocr

    @classmethod
    def get_classifier_model(cls):
        """
        Load the card classifier model (e.g., to distinguish between KTP and NPWP).
        """
        if cls._classifier_model is None:
            # Load your classifier model here
            classifier_model_path = "card_classifier/classifier_ktp_npwp.pth"

            model = models.resnet18(pretrained=False)  # Set to False because you're loading your own weights
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)  # Adjust for binary classification

            model.load_state_dict(torch.load(classifier_model_path))
            model.eval()

            model = model.to(DEVICE)

            cls._classifier_model = model.to(DEVICE)

        return cls._classifier_model

def preprocess_image(image):
    """
    Preprocess an image for inference.

    The pipeline is the same as the one used during training, i.e.,
    resize to (224, 224), convert to tensor, and normalize the image
    with the same mean and standard deviation as the training data.

    Args:
        image (PIL Image): The input image

    Returns:
        image (torch.Tensor): The preprocessed image
    """
    # Step 1: Define the transformation pipeline (same as used during training)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Same normalization as training
    ])

    image = preprocess(image)  # Apply the transformation
    image = image.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
    return image

def classify_card(image, model):
    """
    Classify the given image using the provided model.

    Args:
        image (PIL.Image): The input image.
        model (nn.Module): The card classifier model.

    Returns:
        int: The class index of the predicted class (0 for KTP, 1 for NPWP).
    """
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():  # Disable gradient computation during inference
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)  # Get the class with highest probability

    return preds.item()