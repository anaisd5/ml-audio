import torch
import librosa
import numpy as np
import sys
import json
from .model import get_audio_resnet

# --- Parameters ---
MODEL_PATH = "model_trained.pth"
CLASS_MAP_PATH = "class_map.json"
FIXED_WIDTH = 1280 # should be the same during training


def preprocess_single_file(file_path):
    """
    Function preprocessing one file.

    Args:
        file_path (string): the path to the audio file to process
    
    Returns:
        tensor_input (tensor): a tensor ready for model input
    """

    print(f"Loading and processing the file {file_path}...")
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Calculate the scalogram
        C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'))
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        
        # Manage padding / truncating (as in the dataset)
        current_width = C_db.shape[1]
        if current_width > FIXED_WIDTH:
            C_db = C_db[:, :FIXED_WIDTH]
        else:
            pad_width = FIXED_WIDTH - current_width
            C_db = np.pad(C_db, ((0, 0), (0, pad_width)), mode='constant')
            
        # Format for PyTorch (Batch=1, Channel=1, H, W)
        tensor_input = torch.from_numpy(C_db).float().unsqueeze(0).unsqueeze(0)
        
        return tensor_input

    except Exception as e:
        print(f"Error processing the file: {e}")
        return None


def predict(file_to_predict):
    """
    Principal prediction function.

    Args:
        file_to_predict (string): the path to the audio file to make prediction on
    """
    
    # Loading classes list (from JSON)
    print(f"Loading classes list from {CLASS_MAP_PATH}...")
    try:
        with open(CLASS_MAP_PATH, "r") as f:
            classes = json.load(f)["classes"]
        num_classes = len(classes)
    except FileNotFoundError:
        print(f"Error : File {CLASS_MAP_PATH} not found.")
        print("Please launch train.py first to generate the model.")
        sys.exit(1)
    
    # Load model architecture
    print("Loading the model architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_audio_resnet(num_classes=num_classes).to(device)
    
    # Load the trained weights
    print(f"Loading weights from {MODEL_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: File {MODEL_PATH} not found.")
        print("Please launch train.py first to generate the model.")
        sys.exit(1)
        
    model.eval() # put the model in validation mode

    # Preprocess the audio file
    input_tensor = preprocess_single_file(file_to_predict)
    if input_tensor is None:
        return

    # Make prediction
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output_logits = model(input_tensor)
        probabilities = torch.softmax(output_logits, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()
        
        predicted_class = classes[predicted_index]
        confidence = probabilities[0][predicted_index].item()

    print("\n--- Prediction results ---")
    print(f"File: {file_to_predict}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: poetry run python src/ml_audio/predict.py <path_to_audio_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    predict(file_path)