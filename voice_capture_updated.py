#!/usr/bin/env python3
"""
Windows Voice Capture Client - Updated with timeout fixes and streaming support
Captures voice, sends to WSL bridge, handles streaming responses, and plays TTS
"""

import requests
import json
import time
import sys
import logging
import queue
import threading
from typing import Optional, Dict, Any
import pyaudio
import speech_recognition as sr
import pyttsx3
from pathlib import Path

# Import configuration
try:
    from windows_client_config import *
except ImportError:
    print("Warning: Using default configuration. Create windows_client_config.py for custom settings.")
    # Fallback configuration
    BRIDGE_URL = "http://localhost:7000"
    TTS_URL = "http://localhost:7002"
    CONNECT_TIMEOUT = 5.0
    READ_TIMEOUT = 180.0
    ENABLE_STREAMING = True
    TTS_QUEUE_ENABLED = True

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL if 'LOG_LEVEL' in globals() else 'INFO'),
    format='%(asctime)s.%(msecs)03d — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE if 'LOG_FILE' in globals() else 'voice_client.log'),
        logging.StreamHandler() if LOG_TO_CONSOLE if 'LOG_TO_CONSOLE' in globals() else True else None
    ]
)
logger = logging.getLogger(__name__)


class VoiceCapture:
    """Handles voice capture and wake word detection"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.session_id = f"voice_{int(time.time())}"
        
        # TTS queue for streaming responses
        self.tts_queue = queue.Queue() if TTS_QUEUE_ENABLED else None
        self.tts_thread = None
        
        # Calibrate for ambient noise
        with self.microphone as source:
            logger.info("Calibrating for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def listen_for_audio(self, timeout: Optional[float] = None) -> Optional[sr.AudioData]:
        """Listen for audio input"""
        try:
            with self.microphone as source:
                logger.debug("Listening for audio...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                return audio
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error capturing audio: {e}")
            return None
    
    def transcribe_audio(self, audio: sr.AudioData) -> Optional[str]:
        """Transcribe audio using Google Speech Recognition"""
        try:
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Transcribed: {text}")
            return text.lower()
        except sr.UnknownValueError:
            logger.debug("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return None
    
    def detect_wake_word(self, text: str) -> Optional[str]:
        """Detect wake word in transcribed text"""
        for wake_word, server in WAKE_WORDS.items():
            if wake_word in text:
                logger.info(f"Wake word detected: {wake_word} → {server}")
                return server
        return None
    
    def send_to_bridge_streaming(self, text: str, wake_word: str) -> None:
        """Send to bridge with streaming response support"""
        payload = {
            "text": text,
            "wake_word": wake_word,
            "session_id": self.session_id,
            "stream": True  # Enable streaming
        }
        
        logger.info(f"Sending to bridge (streaming): {text[:50]}...")
        
        try:
            # Start TTS thread if enabled
            if self.tts_queue:
                self.start_tts_thread()
            
            # Make streaming request
            with requests.post(
                f"{BRIDGE_URL}/voice/command",
                json=payload,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            ) as response:
                
                if response.status_code != 200:
                    logger.error(f"Bridge error: {response.status_code}")
                    self.speak_error("Sorry, there was an error processing your request.")
                    return
                
                # Process SSE stream
                for line in response.iter_lines():
                    if line:
                        if line.startswith(b'data: '):
                            try:
                                data = json.loads(line[6:])
                                self.handle_stream_event(data)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON in stream: {e}")
                                continue
                
                # Wait for TTS to finish
                if self.tts_queue:
                    self.tts_queue.put(None)  # Signal end
                    if self.tts_thread:
                        self.tts_thread.join()
                        
        except requests.Timeout:
            logger.error(f"Request timeout after {READ_TIMEOUT}s")
            self.speak_error(TIMEOUT_ERROR_MESSAGE)
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            self.speak_error(CONNECTION_ERROR_MESSAGE)
    
    def send_to_bridge_simple(self, text: str, wake_word: str) -> Optional[Dict[str, Any]]:
        """Send to bridge with simple response (non-streaming)"""
        payload = {
            "text": text,
            "wake_word": wake_word,
            "session_id": self.session_id,
            "stream": False  # Non-streaming
        }
        
        logger.info(f"Sending to bridge (simple): {text[:50]}...")
        
        try:
            response = requests.post(
                f"{BRIDGE_URL}/voice/command",
                json=payload,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)  # Fixed timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Got response from {result.get('server', 'unknown')}")
                
                # Send to TTS
                self.send_to_tts(
                    result.get('response', ''),
                    result.get('voice', DEFAULT_VOICE),
                    result.get('voice_config', {})
                )
                return result
            else:
                logger.error(f"Bridge error: {response.status_code} - {response.text}")
                self.speak_error("Sorry, there was an error processing your request.")
                return None
                
        except requests.Timeout:
            logger.error(f"Request timeout after {READ_TIMEOUT}s")
            self.speak_error(TIMEOUT_ERROR_MESSAGE)
            return None
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            self.speak_error(CONNECTION_ERROR_MESSAGE)
            return None
    
    def handle_stream_event(self, data: Dict[str, Any]) -> None:
        """Handle a streaming event"""
        event_type = data.get("type")
        
        if event_type == "reasoning":
            # Reasoning chunks - speak immediately for feedback
            content = data.get("content", "")
            if len(content) > TTS_MIN_CHUNK_LENGTH if 'TTS_MIN_CHUNK_LENGTH' in globals() else 10:
                logger.debug(f"Reasoning: {content[:50]}...")
                if self.tts_queue:
                    voice_config = data.get("voice_config", {})
                    voice_config["speed"] = voice_config.get("speed", 1.0) * TTS_REASONING_SPEED_MULTIPLIER
                    self.tts_queue.put({
                        "text": content,
                        "voice": data.get("voice", DEFAULT_VOICE),
                        "config": voice_config
                    })
        
        elif event_type == "heartbeat":
            # Heartbeat during long operations
            elapsed = data.get("elapsed", 0)
            if elapsed > HEARTBEAT_SPEAK_THRESHOLD if 'HEARTBEAT_SPEAK_THRESHOLD' in globals() else 10:
                message = f"Still working... {int(elapsed)} seconds"
                logger.info(f"Heartbeat: {message}")
                if self.tts_queue:
                    self.tts_queue.put({
                        "text": message,
                        "voice": data.get("voice", DEFAULT_VOICE),
                        "config": data.get("voice_config", {})
                    })
        
        elif event_type == "message":
            # Final response chunks
            content = data.get("content", "")
            if content:
                logger.debug(f"Message: {content[:50]}...")
                if self.tts_queue:
                    self.tts_queue.put({
                        "text": content,
                        "voice": data.get("voice", DEFAULT_VOICE),
                        "config": data.get("voice_config", {})
                    })
        
        elif event_type == "result":
            # Final complete result
            logger.info("Stream complete")
        
        elif event_type == "error":
            # Error from server
            error_msg = data.get("error", "Unknown error")
            logger.error(f"Server error: {error_msg}")
            self.speak_error(f"Error: {error_msg}")
    
    def start_tts_thread(self) -> None:
        """Start TTS processing thread"""
        if self.tts_thread and self.tts_thread.is_alive():
            return
        
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()
    
    def tts_worker(self) -> None:
        """Worker thread for TTS queue processing"""
        while True:
            try:
                item = self.tts_queue.get(timeout=1)
                if item is None:  # End signal
                    break
                
                self.send_to_tts(
                    item["text"],
                    item["voice"],
                    item.get("config", {})
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")
    
    def send_to_tts(self, text: str, voice: str, config: Dict[str, Any]) -> None:
        """Send text to TTS service"""
        if not text:
            return
        
        payload = {
            "text": text,
            "voice": voice,
            "speed": config.get("speed", DEFAULT_SPEED),
            "pitch": config.get("pitch", DEFAULT_PITCH),
            "play_local": True
        }
        
        try:
            response = requests.post(
                f"{TTS_URL}/speak",
                json=payload,
                timeout=(5, 30)  # TTS should be quick
            )
            if response.status_code != 200:
                logger.error(f"TTS error: {response.status_code}")
        except Exception as e:
            logger.error(f"TTS request error: {e}")
    
    def speak_error(self, message: str) -> None:
        """Speak an error message"""
        logger.warning(f"Speaking error: {message}")
        self.send_to_tts(message, DEFAULT_VOICE, {})
    
    def run(self) -> None:
        """Main loop"""
        logger.info("Voice capture started. Listening for wake words...")
        logger.info(f"Timeouts configured: connect={CONNECT_TIMEOUT}s, read={READ_TIMEOUT}s")
        logger.info(f"Streaming: {'enabled' if ENABLE_STREAMING else 'disabled'}")
        
        print("\n" + "="*60)
        print("VOICE ASSISTANT READY")
        print(f"Wake words: {', '.join(WAKE_WORDS.keys())}")
        print(f"Streaming: {'ON' if ENABLE_STREAMING else 'OFF'}")
        print(f"Timeout: {READ_TIMEOUT}s")
        print("Say 'exit' to quit")
        print("="*60 + "\n")
        
        while True:
            try:
                # Listen for audio
                audio = self.listen_for_audio(timeout=1)
                if not audio:
                    continue
                
                # Transcribe
                text = self.transcribe_audio(audio)
                if not text:
                    continue
                
                # Check for exit
                if "exit" in text or "quit" in text:
                    logger.info("Exit command received")
                    print("Goodbye!")
                    break
                
                # Detect wake word
                wake_word = self.detect_wake_word(text)
                if wake_word:
                    # Remove wake word from text
                    for word in WAKE_WORDS.keys():
                        text = text.replace(word, "").strip()
                    
                    # Send to bridge
                    if ENABLE_STREAMING:
                        self.send_to_bridge_streaming(text, wake_word)
                    else:
                        self.send_to_bridge_simple(text, wake_word)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(1)
        
        logger.info("Voice capture stopped")


def test_connection():
    """Test connection to bridge and TTS services"""
    print("Testing connections...")
    
    # Test bridge
    try:
        response = requests.get(f"{BRIDGE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Bridge: {health['status']}")
            print(f"  Servers: {', '.join(health.get('servers', []))}")
        else:
            print(f"✗ Bridge: Error {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Bridge: {e}")
        return False
    
    # Test TTS
    try:
        response = requests.get(f"{TTS_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ TTS: Online")
        else:
            print(f"✗ TTS: Error {response.status_code}")
    except Exception as e:
        print(f"⚠ TTS: {e} (non-critical)")
    
    return True


if __name__ == "__main__":
    # Test connections first
    if not test_connection():
        print("\nPlease ensure the bridge service is running.")
        print("Start it with: ./start-all.sh (in WSL)")
        sys.exit(1)
    
    print("\nStarting voice capture...\n")
    
    # Run voice capture
    capture = VoiceCapture()
    try:
        capture.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)