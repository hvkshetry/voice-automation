#!/usr/bin/env python3
"""
Deep Thought Voice Router
Simple voice assistant that captures speech and sends to the LLM router.
No pattern matching - the router's LLM handles all intelligence.
"""

import os
import sys
import json
import asyncio
import tomli
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import requests
from datetime import datetime

# Optional imports with helpful error messages
try:
    import openwakeword
    from openwakeword.model import Model as WakeModel
except ImportError:
    print("Please install: pip install openwakeword")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Please install: pip install faster-whisper")
    sys.exit(1)

try:
    from piper import PiperVoice
except ImportError:
    print("Please install: pip install piper-tts")
    sys.exit(1)

try:
    import silero_vad
    silero_vad.load_silero_vad()
    VAD_AVAILABLE = True
except ImportError:
    print("Warning: Silero VAD not available. Install with: pip install silero-vad")
    VAD_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepThought")

class VoiceRouter:
    """Simple voice assistant that routes everything to the LLM."""
    
    def __init__(self, config_path: str = "voice_config.toml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.wake_model = None
        self.whisper_model = None
        self.piper_voice = None
        self.is_listening = False
        self.sample_rate = self.config['system'].get('sample_rate', 16000)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load TOML configuration."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
            
        with open(config_file, 'rb') as f:
            return tomli.load(f)
    
    def initialize(self):
        """Initialize all models."""
        logger.info("Initializing Deep Thought...")
        
        # Initialize wake word detection
        logger.info(f"Loading wake word model for: {self.config['wake']['phrase']}")
        self.wake_model = WakeModel(
            wakeword_models=[self.config['wake']['phrase']],
            inference_framework="onnx"
        )
        
        # Initialize ASR
        logger.info(f"Loading Whisper model: {self.config['asr']['model']}")
        self.whisper_model = WhisperModel(
            self.config['asr']['model'],
            device="cpu",  # Use "cuda" if you have GPU
            compute_type="int8"  # Faster inference
        )
        
        # Initialize TTS
        voice_path = Path(self.config['tts']['piper_voice'])
        if not voice_path.exists():
            logger.warning(f"TTS voice not found: {voice_path}")
            logger.warning("Download from: https://rhasspy.github.io/piper-samples/")
            logger.warning("TTS will be disabled")
        else:
            logger.info(f"Loading TTS voice: {voice_path}")
            self.piper_voice = PiperVoice.load(str(voice_path))
        
        logger.info("Deep Thought is ready. Say 'Deep Thought' to activate.")
    
    def detect_wake_word(self, audio_chunk: np.ndarray) -> bool:
        """Check if wake word is in audio chunk."""
        if self.wake_model is None:
            return False
            
        # Process audio for wake word detection
        prediction = self.wake_model.predict(audio_chunk)
        
        # Check if wake word detected (threshold typically 0.5)
        for word, score in prediction.items():
            if score > 0.5:
                logger.info(f"Wake word detected: {word} (confidence: {score:.2f})")
                return True
        return False
    
    def record_until_silence(self) -> Optional[np.ndarray]:
        """Record audio until silence is detected."""
        logger.info("Listening... (speak now)")
        
        audio_chunks = []
        silence_counter = 0
        max_duration = self.config['voice']['max_recording_duration']
        silence_threshold = self.config['voice']['silence_duration']
        
        # Simple energy-based VAD if Silero not available
        def is_speech(chunk):
            if VAD_AVAILABLE:
                # Use Silero VAD
                return True  # Implement Silero check
            else:
                # Simple energy threshold
                energy = np.sqrt(np.mean(chunk**2))
                return energy > 0.01
        
        start_time = datetime.now()
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1) as stream:
            while True:
                chunk, _ = stream.read(int(self.sample_rate * 0.1))  # 100ms chunks
                audio_chunks.append(chunk)
                
                # Check for speech
                if not is_speech(chunk.flatten()):
                    silence_counter += 0.1
                    if silence_counter >= silence_threshold:
                        logger.info("Silence detected, processing...")
                        break
                else:
                    silence_counter = 0
                
                # Check max duration
                if (datetime.now() - start_time).seconds > max_duration:
                    logger.info("Max recording duration reached")
                    break
        
        if audio_chunks:
            return np.concatenate(audio_chunks)
        return None
    
    def transcribe_audio(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper."""
        if self.whisper_model is None:
            return None
            
        # Transcribe
        segments, _ = self.whisper_model.transcribe(
            audio,
            language=self.config['asr'].get('language', 'en'),
            vad_filter=True
        )
        
        # Combine segments
        text = " ".join([seg.text.strip() for seg in segments])
        
        if text:
            logger.info(f"Transcribed: {text}")
            return text
        return None
    
    def call_mcp_router(self, prompt: str) -> Optional[str]:
        """Send prompt to MCP router and get response."""
        endpoint = self.config['mcp']['endpoint']
        timeout = self.config['mcp']['timeout']
        
        # Prepare MCP tool call
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "codex",
                "arguments": {
                    "prompt": prompt,
                    "approval-policy": "never",
                    "sandbox": "workspace-write",
                    "config": {
                        "sandbox_workspace_write": {
                            "network_access": True
                        }
                    }
                }
            }
        }
        
        logger.info(f"Sending to router: {prompt[:100]}...")
        
        try:
            # Send to MCP endpoint
            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response text from MCP result
                if 'result' in result:
                    return result['result'].get('content', [{}])[0].get('text', "Done")
                return "Request processed"
            else:
                logger.error(f"MCP error: {response.status_code}")
                return "Sorry, there was an error processing your request"
                
        except requests.RequestException as e:
            logger.error(f"Network error: {e}")
            return "Sorry, I couldn't connect to the router"
    
    def speak(self, text: str):
        """Speak text using Piper TTS."""
        if self.piper_voice is None:
            logger.info(f"TTS disabled. Would say: {text}")
            return
            
        logger.info(f"Speaking: {text[:100]}...")
        
        # Generate audio
        audio_data = self.piper_voice.synthesize(
            text,
            rate=self.config['tts'].get('speech_rate', 1.0)
        )
        
        # Play audio
        sd.play(audio_data, samplerate=22050)  # Piper typically outputs 22050 Hz
        sd.wait()
    
    async def run(self):
        """Main event loop."""
        self.initialize()
        
        logger.info("Starting audio stream...")
        
        # Continuous listening loop
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
        ) as stream:
            
            while True:
                try:
                    # Read audio chunk
                    audio_chunk, _ = stream.read(int(self.sample_rate * 0.5))
                    
                    # Check for wake word
                    if self.detect_wake_word(audio_chunk.flatten()):
                        # Play acknowledgment sound or speak
                        self.speak("Yes?")
                        
                        # Record user's request
                        audio = self.record_until_silence()
                        
                        if audio is not None:
                            # Transcribe
                            text = self.transcribe_audio(audio)
                            
                            if text:
                                # Send to router - NO PATTERN MATCHING!
                                response = self.call_mcp_router(text)
                                
                                if response:
                                    # Speak response
                                    self.speak(response)
                            else:
                                self.speak("I didn't catch that.")
                    
                    # Small delay to prevent CPU spinning
                    await asyncio.sleep(0.01)
                    
                except KeyboardInterrupt:
                    logger.info("Shutting down Deep Thought...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(1)

def main():
    """Entry point."""
    print("""
    ╔══════════════════════════════════════╗
    ║       DEEP THOUGHT VOICE ROUTER      ║
    ║   Say 'Deep Thought' to activate     ║
    ║         Press Ctrl+C to exit          ║
    ╚══════════════════════════════════════╝
    """)
    
    # Check for config file argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "voice_config.toml"
    
    # Create and run router
    router = VoiceRouter(config_path)
    
    # Run async event loop
    try:
        asyncio.run(router.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()