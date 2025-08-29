#!/usr/bin/env python3
"""
Download the missing Silero VAD model for OpenWakeWord
"""

import os
import requests
from pathlib import Path

def download_vad_model():
    # URL for the Silero VAD model - from openWakeWord GitHub releases
    # This is the correct URL as specified in openwakeword/__init__.py
    model_url = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx"
    
    # Get the target path from the package
    import openwakeword
    package_dir = Path(openwakeword.__file__).parent
    models_dir = package_dir / "resources" / "models"
    target_path = models_dir / "silero_vad.onnx"
    
    print(f"Package directory: {package_dir}")
    print(f"Target path: {target_path}")
    
    # Create directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the model
    print(f"Downloading VAD model from {model_url}...")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    
    # Save the model
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"Progress: {percent:.1f}%", end='\r')
    
    print(f"\nVAD model downloaded successfully to {target_path}")
    print(f"File size: {os.path.getsize(target_path):,} bytes")

if __name__ == "__main__":
    try:
        download_vad_model()
    except Exception as e:
        print(f"Error downloading VAD model: {e}")
        print("\nYou can manually download it from:")
        print("https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx")
        print("And place it in the openwakeword/resources/models/ directory")