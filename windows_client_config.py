"""
Windows Voice Client Configuration
Configure timeouts, streaming, and other client settings
"""

# Connection Configuration
BRIDGE_URL = "http://localhost:7000"  # WSL Bridge endpoint
TTS_URL = "http://localhost:7002"     # Local Windows TTS service

# Timeout Configuration (in seconds)
CONNECT_TIMEOUT = 5.0   # Time to establish connection
READ_TIMEOUT = 180.0    # Time to wait for response (3 minutes for long MCP operations)

# Previous timeout that caused issues
# OLD_TIMEOUT = 30.0  # This was too short for Office MCP (~63s operations)

# Streaming Configuration
ENABLE_STREAMING = True  # Enable SSE streaming for progressive responses
STREAM_CHUNK_SIZE = 1024  # Bytes per chunk when streaming
STREAM_RECONNECT_DELAY = 2.0  # Seconds before reconnecting on stream failure

# TTS Configuration
TTS_QUEUE_ENABLED = True  # Enable TTS queue for streaming audio
TTS_REASONING_SPEED_MULTIPLIER = 1.1  # Speed up reasoning narration slightly
TTS_MIN_CHUNK_LENGTH = 10  # Minimum characters before sending to TTS
TTS_SENTENCE_BOUNDARIES = True  # Wait for sentence boundaries before speaking

# Heartbeat Configuration
HEARTBEAT_SPEAK_THRESHOLD = 10  # Speak heartbeat messages after N seconds of silence
HEARTBEAT_MESSAGE_TEMPLATE = "Still working... {elapsed} seconds"

# Error Messages
TIMEOUT_ERROR_MESSAGE = "The request is taking longer than expected. Still waiting..."
CONNECTION_ERROR_MESSAGE = "Could not connect to the AI assistant. Please check the connection."
STREAM_ERROR_MESSAGE = "Stream interrupted. Reconnecting..."

# Voice Detection
WAKE_WORDS = {
    "jarvis": "router",
    "office": "office-assistant", 
    "analyst": "openbb-analyst",
    "router": "router"
}

# Default Voice Settings
DEFAULT_VOICE = "en-US-ChristopherNeural"
DEFAULT_SPEED = 1.0
DEFAULT_PITCH = 1.0

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "voice_client.log"
LOG_TO_CONSOLE = True

# Performance Tuning
MAX_RETRIES = 3  # Maximum retry attempts for failed requests
RETRY_DELAY = 1.0  # Seconds between retries
CHUNK_PROCESSING_DELAY = 0.01  # Small delay between processing chunks (seconds)

# Feature Flags
TWO_STAGE_DETECTION = True  # Enable two-stage wake word detection
SAVE_AUDIO_CLIPS = False  # Save audio clips for debugging
SHOW_REASONING = True  # Display/speak reasoning chunks
SHOW_HEARTBEATS = True  # Display/speak heartbeat messages