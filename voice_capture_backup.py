#!/usr/bin/env python3
"""
Windows Voice Capture with Multiple Wake Words
Captures audio and routes to different specialists
"""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import tomli
import numpy as np
import sounddevice as sd
import requests
from openwakeword.model import Model
import queue
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parent / "config.toml"
if config_path.exists():
    with open(config_path, "rb") as f:
        config = tomli.load(f)
else:
    # Default configuration
    config = {
        "bridge": {
            "url": "http://localhost:7000",
            "timeout": 180  # Increased from 30s to 3 minutes for long MCP operations
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_duration": 1.0,
            "silence_threshold": 0.01,
            "silence_duration": 1.5,
            "max_recording_duration": 30.0
        },
        "wake_words": {
            "deep_thought": {
                "model": "alexa",  # Using alexa as placeholder
                "threshold": 0.5,
                "target": "router",
                "response": "Hmm, let me ponder that..."
            },
            "hey_office": {
                "model": "hey_jarvis",
                "threshold": 0.5,
                "target": "office",
                "response": "Checking your office..."
            },
            "hey_analyst": {
                "model": "hey_jarvis",
                "threshold": 0.5,
                "target": "analyst",
                "response": "Analyzing the markets..."
            }
        },
        "tts": {
            "enabled": True,
            "url": "http://localhost:7002",
            "voice": "en_GB-alan-medium"
        }
    }

class VoiceCapture:
    def __init__(self):
        self.config = config
        self.audio_queue = queue.Queue()
        self.wake_word_detected = None
        self.recording = False
        self.recorded_audio = []
        self.is_speaking = False  # Flag to prevent detection during TTS playback
        self.last_detection_time = 0  # For debounce protection
        self.debounce_time = 2.0  # Seconds to wait before allowing new detection
        self.consecutive_silent_chunks = 0  # Track consecutive silent chunks
        self.required_silent_chunks = 20  # ~1.6 seconds at 80ms/chunk
        self.min_recording_chunks = 13  # ~1 second minimum before checking silence
        self.frame_count = 0  # Track frames processed for warm-up period
        self.recording_start_time = 0  # Track when recording actually started
        self.discard_until = 0  # Timestamp to discard audio until (for post-TTS cleanup)
        
        # Pre-recording buffer to capture audio before wake word
        from collections import deque
        self.pre_recording_buffer = deque(maxlen=10)  # Keep last 10 chunks (~800ms)
        
        # Initialize wake word detection with local models
        models_path = Path(__file__).parent / "models" / "wakewords"
        self.wake_model = Model(
            wakeword_models=[
                str(models_path / "alexa_v0.1.onnx"),
                str(models_path / "hey_jarvis_v0.1.onnx")
            ],
            inference_framework="onnx",
            melspec_model_path=str(models_path / "melspectrogram.onnx"),
            embedding_model_path=str(models_path / "embedding_model.onnx"),
            vad_threshold=0.3  # Enable VAD with 30% confidence threshold
        )
        
        # Audio stream parameters
        self.sample_rate = config["audio"]["sample_rate"]
        self.chunk_size = int(self.sample_rate * config["audio"]["chunk_duration"])
        
        logger.info("Voice capture initialized")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Monitor audio level
        audio_level = np.abs(indata).mean()
        if audio_level > 0.005:  # Show when there's significant audio
            logger.debug(f"Audio detected - level: {audio_level:.4f}")
        
        # Add to queue with timestamp for processing
        current_time = time.time()
        self.audio_queue.put((current_time, indata.copy()))
    
    def process_audio(self):
        """Process audio chunks for wake word detection"""
        while True:
            try:
                # Get audio chunk with timestamp
                chunk_time, audio_chunk = self.audio_queue.get(timeout=1)
                
                # Discard old audio from before TTS cleanup time
                if chunk_time < self.discard_until:
                    logger.debug(f"Discarding old chunk from {chunk_time:.2f} (before {self.discard_until:.2f})")
                    continue
                
                if self.recording:
                    # Add to recording buffer
                    self.recorded_audio.append(audio_chunk)
                    # Only log every 10th chunk to reduce noise
                    if len(self.recorded_audio) % 10 == 0:
                        logger.debug(f"Recording... {len(self.recorded_audio)} chunks captured")
                    
                    # Only check for silence after minimum recording duration
                    if len(self.recorded_audio) >= self.min_recording_chunks:
                        if self.check_silence(audio_chunk):
                            self.consecutive_silent_chunks += 1
                            logger.debug(f"Silent chunk {self.consecutive_silent_chunks}/{self.required_silent_chunks}")
                            
                            # Stop only after enough consecutive silent chunks
                            if self.consecutive_silent_chunks >= self.required_silent_chunks:
                                self.stop_recording()
                        else:
                            # Reset counter if we hear sound
                            if self.consecutive_silent_chunks > 0:
                                logger.debug("Sound detected, resetting silence counter")
                            self.consecutive_silent_chunks = 0
                else:
                    # Add to pre-recording buffer when not recording
                    self.pre_recording_buffer.append(audio_chunk)
                    
                    if not self.is_speaking:
                        # Only check for wake words when not speaking
                        self.detect_wake_word(audio_chunk)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {str(e)}")
    
    def detect_wake_word(self, audio_chunk):
        """Detect wake words in audio chunk"""
        # Convert to 16-bit PCM format expected by OpenWakeWord
        # sounddevice gives us float32 in range [-1, 1], convert to int16
        audio_int16 = (audio_chunk * 32767).astype(np.int16).flatten()
        
        # Increment frame counter
        self.frame_count += 1
        
        # Debug audio format once
        if not hasattr(self, '_logged_audio_format'):
            logger.debug(f"Audio format - shape: {audio_int16.shape}, dtype: {audio_int16.dtype}, range: [{audio_int16.min()}, {audio_int16.max()}]")
            logger.debug(f"Frame {self.frame_count}: Starting wake word detection (need 5+ frames for warm-up)")
            self._logged_audio_format = True
        
        # Get predictions - first without patience to see raw scores
        predictions = self.wake_model.predict(audio_int16)
        
        # Debug: Log if predictions is empty or what keys it has
        if not predictions:
            logger.debug("No predictions returned from model")
            return
        
        # Show prediction keys on first call for debugging
        if not hasattr(self, '_logged_keys'):
            logger.info(f"Prediction dictionary keys: {list(predictions.keys())}")
            self._logged_keys = True
        
        # Debug: Show ALL prediction values to see what's happening
        for key, value in predictions.items():
            # Always show the score for debugging
            if "/" in key or "\\" in key:
                model_name = Path(key).stem
            else:
                model_name = key
            
            # Show all scores when they change or during warm-up
            if value > 0.0:
                logger.debug(f"Frame {self.frame_count} - Model '{model_name}': score={value:.4f}")
            elif self.frame_count <= 5:
                logger.debug(f"Frame {self.frame_count} - Model '{model_name}': score={value:.4f} (warm-up)")
            elif self.frame_count == 6 and not hasattr(self, '_warmup_done'):
                logger.info("Warm-up complete! Wake word detection active.")
                self._warmup_done = True
        
        # Check each configured wake word
        for wake_name, wake_config in self.config["wake_words"].items():
            model_name = wake_config["model"]
            threshold = wake_config["threshold"]
            
            # The actual key format used by OpenWakeWord is model_name_v0.1
            prediction_key = f"{model_name}_v0.1"
            
            if prediction_key in predictions:
                score = predictions[prediction_key]
                
                # Always log ALL scores for debugging
                if score > 0.0:  # Log ANY score
                    logger.debug(f"Wake word '{wake_name}' score: {score:.4f} (threshold: {threshold})")
                
                if score > threshold:
                    # Check debounce time
                    current_time = time.time()
                    if current_time - self.last_detection_time < self.debounce_time:
                        logger.debug(f"Debounce: Ignoring detection within {self.debounce_time}s")
                        return
                    
                    logger.info(f"🎙️ DETECTED: {wake_name} (score: {score:.2f})")
                    self.wake_word_detected = wake_name
                    self.last_detection_time = current_time
                    
                    # Play a quick chime instead of TTS
                    self.play_chime()
                    
                    self.start_recording()
                    return  # Exit after first detection
    
    def play_chime(self):
        """Play a quick chime sound to acknowledge wake word detection"""
        try:
            # Generate a simple sine wave chime (440Hz for 200ms)
            duration = 0.2  # seconds
            frequency = 440  # Hz (A4 note)
            samples = int(self.sample_rate * duration)
            t = np.linspace(0, duration, samples)
            
            # Create sine wave with fade in/out to avoid clicks
            chime = np.sin(2 * np.pi * frequency * t)
            
            # Apply envelope (fade in and out)
            fade_samples = int(0.01 * self.sample_rate)  # 10ms fade
            chime[:fade_samples] *= np.linspace(0, 1, fade_samples)
            chime[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Reduce volume
            chime *= 0.3
            
            # Play the chime
            sd.play(chime, self.sample_rate)
            # Don't wait - let recording start immediately
            logger.debug("Chime played")
        except Exception as e:
            logger.error(f"Failed to play chime: {e}")
    
    def check_silence(self, audio_chunk) -> bool:
        """Check if audio chunk is silence"""
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_chunk**2))
        return rms < self.config["audio"]["silence_threshold"]
    
    def start_recording(self):
        """Start recording after wake word"""
        self.recording = True
        
        # Start with pre-recording buffer to capture speech before wake word
        self.recorded_audio = list(self.pre_recording_buffer)
        logger.info(f"Started recording with {len(self.recorded_audio)} pre-buffer chunks")
        
        self.consecutive_silent_chunks = 0  # Reset silence counter
        self.recording_start_time = time.time()
        logger.info(f"Recording started at {self.recording_start_time:.2f}")
    
    def stop_recording(self):
        """Stop recording and process"""
        if not self.recording:
            return
        
        self.recording = False
        duration = time.time() - self.recording_start_time
        num_chunks = len(self.recorded_audio)
        logger.info(f"Recording stopped (duration: {duration:.1f}s, total chunks: {num_chunks})")
        
        if self.recorded_audio:
            # Combine audio chunks
            audio_data = np.concatenate(self.recorded_audio)
            total_samples = len(audio_data)
            actual_duration = total_samples / self.sample_rate
            
            # Calculate expected duration (including pre-buffer)
            chunk_duration = 0.08  # 80ms per chunk
            expected_duration = num_chunks * chunk_duration
            
            logger.info(f"Audio buffer: {total_samples} samples, {actual_duration:.2f}s")
            logger.info(f"Expected from chunks: {expected_duration:.2f}s ({num_chunks} chunks * {chunk_duration}s)")
            
            # Check audio levels to see if we captured speech
            audio_level = np.abs(audio_data).mean()
            logger.info(f"Average audio level: {audio_level:.4f}")
            
            # Convert to WAV format
            import io
            import wave
            
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                # Convert to int16 and write
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Get WAV size for debugging
            wav_buffer.seek(0)
            wav_data = wav_buffer.read()
            logger.debug(f"WAV file size: {len(wav_data)} bytes")
            
            # Encode as base64
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            logger.debug(f"Base64 length: {len(audio_base64)}")
            
            # Send to bridge service
            self.send_to_bridge(audio_base64)
        
        # Clear the buffer
        self.recorded_audio = []
        self.wake_word_detected = None
    
    def send_to_bridge(self, audio_base64: str):
        """Send audio to WSL bridge service"""
        if not self.wake_word_detected:
            return
        
        wake_config = self.config["wake_words"][self.wake_word_detected]
        target = wake_config["target"]
        
        try:
            # Send to bridge
            response = requests.post(
                f"{self.config['bridge']['url']}/voice/command",
                json={
                    "audio_data": audio_base64,
                    "wake_word": target,
                    "session_id": str(time.time())
                },
                timeout=self.config["bridge"]["timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                logger.info(f"Response: {response_text[:100]}...")
                
                # Speak response if TTS enabled
                if self.config["tts"]["enabled"] and response_text:
                    self.speak(response_text)
            else:
                logger.error(f"Bridge error: {response.status_code} - {response.text}")
                if self.config["tts"]["enabled"]:
                    self.speak("Sorry, I encountered an error processing your request.")
        
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            if self.config["tts"]["enabled"]:
                self.speak("The request timed out. Please try again.")
        
        except Exception as e:
            logger.error(f"Bridge communication error: {str(e)}")
            if self.config["tts"]["enabled"]:
                self.speak("I couldn't connect to the processing service.")
    
    def speak(self, text: str):
        """Send text to TTS service"""
        if not text:
            return
        
        self.is_speaking = True  # Disable wake word detection
        
        try:
            response = requests.post(
                f"{self.config['tts']['url']}/speak",
                json={
                    "text": text,
                    "voice": self.config["tts"]["voice"]
                },
                timeout=10
            )
            
            if response.status_code == 200:
                # Play audio response
                audio_data = response.content
                
                # Play the WAV audio using sounddevice
                import io
                import soundfile as sf
                
                try:
                    # Read WAV data
                    audio_io = io.BytesIO(audio_data)
                    data, samplerate = sf.read(audio_io)
                    
                    # Play audio and wait for completion
                    sd.play(data, samplerate)
                    sd.wait()  # Wait until audio finishes playing
                    
                    logger.info("TTS response played")
                except Exception as e:
                    logger.error(f"Failed to play audio: {e}")
            else:
                logger.error(f"TTS error: {response.status_code}")
        
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
        finally:
            # Clear audio queue AFTER TTS completes
            cleared_count = 0
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    cleared_count += 1
                except:
                    break
            if cleared_count > 0:
                logger.debug(f"Cleared {cleared_count} audio chunks after TTS")
            
            # Reset the wake word model to clear prediction buffers
            self.wake_model.reset()
            logger.debug("Reset wake word model after TTS")
            
            # Set timestamp to discard any remaining old audio
            self.discard_until = time.time() + 0.5  # Discard audio for next 500ms
            
            self.is_speaking = False  # Re-enable wake word detection
    
    def run(self):
        """Main run loop"""
        logger.info("Starting voice capture...")
        logger.info(f"Wake words: {', '.join(self.config['wake_words'].keys())}")
        logger.info(f"Bridge URL: {self.config['bridge']['url']}")
        
        # Start audio processing thread
        process_thread = threading.Thread(target=self.process_audio, daemon=True)
        process_thread.start()
        
        # Start audio stream
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.chunk_size
            ):
                logger.info("Listening for wake words...")
                
                # Keep running
                while True:
                    time.sleep(1)
                    
                    # Check if recording timeout
                    if self.recording:
                        duration = time.time() - self.recording_start_time
                        if duration > self.config["audio"]["max_recording_duration"]:
                            logger.warning("Recording timeout")
                            self.stop_recording()
        
        except KeyboardInterrupt:
            logger.info("Stopping voice capture...")
        except Exception as e:
            logger.error(f"Audio stream error: {str(e)}")

def main():
    """Main entry point"""
    import sys
    
    # Enable debug logging if requested
    if "--debug" in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
        print("Debug logging enabled - you'll see audio levels and wake word scores")
    
    # Test microphone if requested
    if "--test-mic" in sys.argv:
        print("\n🎤 Testing microphone for 5 seconds...")
        print("Speak now and watch the audio levels:\n")
        
        def test_callback(indata, frames, time_info, status):
            level = np.abs(indata).mean()
            bars = "█" * int(level * 500)  # Visual level meter
            if level > 0.001:
                print(f"\rAudio Level: {level:.4f} {bars}", end="", flush=True)
        
        with sd.InputStream(callback=test_callback, channels=1, samplerate=16000):
            time.sleep(5)
        
        print("\n\nMicrophone test complete!")
        return
    
    # Show help if requested
    if "--help" in sys.argv:
        print("\nVoice Capture Options:")
        print("  --debug      Enable debug logging to see audio levels and scores")
        print("  --test-mic   Test microphone input for 5 seconds")
        print("  --help       Show this help message\n")
        return
    
    capture = VoiceCapture()
    capture.run()

if __name__ == "__main__":
    main()