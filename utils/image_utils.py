import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
MODEL_PATH = "asl_model.h5"
model = load_model(MODEL_PATH)

# Define class labels (A-Z, SPACE, DELETE, NOTHING)
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'SPACE', 'DELETE', 'NOTHING'
]

def preprocess_image(img_path, target_size=(64, 64)):
    """
    Preprocess the image for prediction.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_asl(img_path):
    """
    Predict the ASL sign from an image.
    """
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return CLASS_LABELS[predicted_class[0]]