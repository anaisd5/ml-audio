import librosa
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Files definition
SOURCE_DIR = Path("data/gtzan/audio/") 
TARGET_DIR = Path("data/processed/scalograms/")

# Check if the output directory exist and if not create it with parents
TARGET_DIR.mkdir(parents=True, exist_ok=True)


def process_file(file_path, target_path):
    """
    Load an audio file, transform it in a CQT scalogram 
    and save it in a Numpy array.

    Args:
        file_path (string): the path to the audio file to process
        target_path (string): the path to save the output file
    """

    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None) # y is the audio signal, sr is the sample rate
        
        # Do the Constant Q Transform
        C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1')) # fmin is a filter: only consider notes higher than C1
                                                                # C is a 2D Numpy array with complex numbers
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)   # only consider the amplitude of the signal (not the phase)
                                                                # convert into dB 
                                                                # (negative since the reference is max amplitude of the signal)
        
        # Save raw data (Numpy array) at target_path
        np.save(target_path, C_db)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)


# Main code
if __name__ == "__main__":
    print("Starting preprocessing...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {TARGET_DIR}")

    # Find all audio files
    audio_files = list(SOURCE_DIR.rglob("*.wav")) # .rglob("*.wav") searches recursively in all subfolders
    
    if not audio_files:
        print(f"Error : No .wav file found in {SOURCE_DIR}")
        sys.exit(1)
        
    print(f"{len(audio_files)} audio files found.")

    # Main loop for preprocessing
    for file_path in tqdm(audio_files, desc="Files preprocessing"): # tqdm adds a progress bar
        
        # Create the output path keeping the folder structure
        # e.g. data/gtzan/audio/blues/file.wav -> data/processed/scalograms/blues/file.npy
        
        # Calculate the relative path (e.g. 'blues/file.wav')
        relative_path = file_path.relative_to(SOURCE_DIR)
        
        # Create the destination path (e.g. 'data/processed/scalograms/blues/file.npy')
        target_path = TARGET_DIR / relative_path.with_suffix(".npy")
        
        # Create the parent folder if necessary (ex: 'data/processed/scalograms/blues/')
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Call the function to process the file
        process_file(file_path, target_path)

    print("Preprocessing done.")