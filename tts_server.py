#!/usr/bin/env python3
"""
Piper TTS Server
Provides text-to-speech using Piper with British voice
"""

import asyncio
import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import sounddevice as sd
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Piper TTS Server")

# Voice models directory
VOICES_DIR = Path(__file__).parent / "voices"

# Available voices
VOICES = {
    "en_GB-alan-medium": {
        "model": VOICES_DIR / "en_GB-alan-medium.onnx",
        "config": VOICES_DIR / "en_GB-alan-medium.onnx.json",
        "description": "British male voice (Stephen Fry-like)"
    },
    "en_GB-jenny_dioco-medium": {
        "model": VOICES_DIR / "en_GB-jenny_dioco-medium.onnx",
        "config": VOICES_DIR / "en_GB-jenny_dioco-medium.onnx.json",
        "description": "British female voice"
    },
    "en_US-joe-medium": {
        "model": VOICES_DIR / "en_US-joe-medium.onnx",
        "config": VOICES_DIR / "en_US-joe-medium.onnx.json",
        "description": "American male voice"
    }
}

class TTSRequest(BaseModel):
    """TTS request model"""
    text: str
    voice: str = "en_GB-alan-medium"
    play_local: bool = False  # Play on server's audio device
    format: str = "wav"  # wav or mp3

def check_piper_installed() -> bool:
    """Check if Piper is installed"""
    try:
        result = subprocess.run(
            ["piper", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def synthesize_speech(text: str, voice: str = "en_GB-alan-medium") -> bytes:
    """
    Synthesize speech using Piper
    
    Args:
        text: Text to synthesize
        voice: Voice model to use
    
    Returns:
        WAV audio data as bytes
    """
    if voice not in VOICES:
        raise ValueError(f"Unknown voice: {voice}")
    
    voice_config = VOICES[voice]
    model_path = voice_config["model"]
    config_path = voice_config["config"]
    
    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Voice model not found: {model_path}")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Run Piper
        cmd = [
            "piper",
            "--model", str(model_path),
            "--config", str(config_path),
            "--output_file", tmp_path
        ]
        
        logger.info(f"Synthesizing: {text[:50]}...")
        
        result = subprocess.run(
            cmd,
            input=text,
            text=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            logger.error(f"Piper error: {result.stderr}")
            raise RuntimeError(f"Piper synthesis failed: {result.stderr}")
        
        # Read the generated audio
        with open(tmp_path, "rb") as f:
            audio_data = f.read()
        
        return audio_data
    
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.on_event("startup")
async def startup_event():
    """Check dependencies on startup"""
    if not check_piper_installed():
        logger.warning("Piper not installed! Install with: pip install piper-tts")
    
    # Check for at least one voice model
    found_voice = False
    for voice_name, voice_config in VOICES.items():
        if voice_config["model"].exists():
            logger.info(f"Found voice model: {voice_name}")
            found_voice = True
    
    if not found_voice:
        logger.warning("No voice models found! Download models to voices/ directory")

@app.post("/speak")
async def speak(request: TTSRequest):
    """
    Convert text to speech
    
    Args:
        request: TTS request with text and voice selection
    
    Returns:
        Audio file response or plays locally
    """
    try:
        # Synthesize speech
        audio_data = synthesize_speech(request.text, request.voice)
        
        if request.play_local:
            # Play on local audio device
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            try:
                # Load and play audio
                data, samplerate = sf.read(tmp_path)
                sd.play(data, samplerate)
                sd.wait()  # Wait until playback is finished
                
                return {"status": "played", "voice": request.voice}
            
            finally:
                os.unlink(tmp_path)
        
        else:
            # Return audio file
            return Response(
                content=audio_data,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=speech.wav"
                }
            )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speak/stream")
async def speak_stream(request: TTSRequest):
    """
    Stream synthesized speech
    """
    try:
        # Synthesize speech
        audio_data = synthesize_speech(request.text, request.voice)
        
        # Stream the audio
        async def audio_streamer():
            chunk_size = 4096
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i + chunk_size]
        
        return StreamingResponse(
            audio_streamer(),
            media_type="audio/wav"
        )
    
    except Exception as e:
        logger.error(f"Stream TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def list_voices():
    """
    List available voices
    """
    available = []
    for voice_name, voice_config in VOICES.items():
        if voice_config["model"].exists():
            available.append({
                "id": voice_name,
                "description": voice_config["description"],
                "available": True
            })
        else:
            available.append({
                "id": voice_name,
                "description": voice_config["description"],
                "available": False,
                "note": "Model not downloaded"
            })
    
    return {"voices": available}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    piper_installed = check_piper_installed()
    voices_available = any(
        voice_config["model"].exists() 
        for voice_config in VOICES.values()
    )
    
    return {
        "status": "healthy" if piper_installed and voices_available else "degraded",
        "piper_installed": piper_installed,
        "voices_available": voices_available
    }

if __name__ == "__main__":
    # Create voices directory if it doesn't exist
    VOICES_DIR.mkdir(exist_ok=True)
    
    # Download instructions
    if not any(v["model"].exists() for v in VOICES.values()):
        print("\n" + "="*50)
        print("No voice models found!")
        print("\nDownload the British voice (Deep Thought):")
        print("  cd voices")
        print("  wget https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-gb-alan-medium.tar.gz")
        print("  tar -xzf voice-en-gb-alan-medium.tar.gz")
        print("="*50 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=7002,
        log_level="info"
    )