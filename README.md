# Voice Automation for Codex MCP Orchestra

A voice interface component that enables natural speech interaction with the Codex MCP Orchestra multi-agent system. This module provides wake word detection, speech recognition, and text-to-speech capabilities.

## Overview

This voice automation system acts as a hands-free interface to your AI agents, allowing you to:
- Activate agents using customizable wake words
- Speak naturally to interact with AI assistants
- Receive spoken responses with configurable voices
- Route requests to appropriate specialized agents

## Architecture

```
Voice Input → Wake Word Detection → Speech Recognition → MCP Router
                                                            ↓
Voice Output ← Text-to-Speech ← Response Processing ← AI Agent Response
```

## Features

- **Wake Word Detection**: Uses OpenWakeWord for customizable activation phrases
- **Speech Recognition**: Faster-whisper for accurate, efficient transcription
- **Voice Activity Detection**: Silero VAD for improved speech detection
- **Text-to-Speech**: Piper TTS with multiple voice options
- **MCP Integration**: Direct connection to Codex MCP servers
- **Async Processing**: Non-blocking audio stream handling

## Prerequisites

### System Requirements
- Python 3.9+
- Audio input device (microphone)
- Audio output device (speakers/headphones)
- Linux/WSL2 or Windows with proper audio routing

### Core Dependencies
- OpenWakeWord for wake word detection
- Faster-whisper for speech recognition
- Piper TTS for voice synthesis
- SoundDevice for audio I/O

## Installation

### 1. Clone Repository (Standalone)
```bash
git clone https://github.com/hvkshetry/voice-automation.git
cd voice-automation
```

Or as part of Codex MCP Orchestra:
```bash
git clone --recurse-submodules https://github.com/hvkshetry/codex-mcp-orchestra.git
```

### 2. Install Dependencies
```bash
pip install openwakeword>=0.5.1
pip install faster-whisper>=0.10.0
pip install piper-tts>=1.2.0
pip install sounddevice>=0.4.6
pip install numpy>=1.24.0
pip install silero-vad>=3.1.0  # Optional but recommended
pip install httpx tomli  # For configuration and HTTP requests
```

### 3. Download Voice Models

#### Whisper Models
Models download automatically on first use. Available sizes:
- `tiny` - Fastest, least accurate
- `base` - Good balance for testing
- `small` - Recommended for production
- `medium` - Better accuracy, slower
- `large` - Best accuracy, requires GPU

#### TTS Voices
Download Piper voice models:
```bash
mkdir -p voices
cd voices

# British male voice (recommended for "Deep Thought" persona)
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_GB-alan-medium.onnx
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_GB-alan-medium.onnx.json

# Alternative voices
# US female: en_US-amy-medium.onnx
# US male: en_US-ryan-medium.onnx
```

## Configuration

Edit `voice_config.toml`:

```toml
[wake]
# Wake phrase - customize to your preference
phrase = "deep thought"
# Alternative phrases:
# phrase = "hey assistant"
# phrase = "computer"

[asr]
# Whisper model size
model = "small"  # tiny, base, small, medium, large
language = "en"  # Language code for better accuracy

[tts]
# Path to Piper voice model
piper_voice = "voices/en_GB-alan-medium.onnx"
speech_rate = 0.9  # 1.0 = normal, 0.8 = slower, 1.2 = faster

[mcp]
# MCP server endpoint
endpoint = "http://127.0.0.1:8090/sse"
timeout = 600  # Request timeout in seconds (10 minutes for complex operations)

[voice]
# Voice activity detection settings
vad_threshold = 0.5  # Confidence threshold (0-1)
silence_duration = 1.5  # Seconds of silence before end of speech
max_recording_duration = 30  # Maximum recording length

[system]
# Audio configuration
sample_rate = 16000
# input_device_index = 0  # Uncomment to specify device
# output_device_index = 0  # Uncomment to specify device
```

## Usage

### Standalone Mode
```bash
python voice_router.py
```

### With Codex MCP Orchestra
Ensure MCP servers are running first:
```bash
# In main orchestra directory
./start-all.sh

# Then start voice service
cd voice-automation
python voice_router.py
```

### Interaction Flow

1. **Activation**: Say the wake word (e.g., "Deep Thought")
2. **Listen Indicator**: System plays a chime or shows "Listening..."
3. **Speak**: Say your request naturally
4. **Processing**: System transcribes and sends to AI
5. **Response**: AI response is spoken back to you

### Example Interactions

```
You: "Deep Thought"
System: *chime*
You: "What's the weather like today?"
System: "I'll help you check the weather..."

You: "Deep Thought"
System: *chime*
You: "Schedule a meeting for tomorrow at 3pm"
System: "I'll schedule that meeting for you..."
```

## Customization

### Custom Wake Words

OpenWakeWord supports custom wake word training. For now, use pre-trained models:
- "alexa"
- "hey jarvis"
- "hey mycroft"
- "ok google"

### Voice Personalities

Configure different voices for different agents in the main config:
```python
# In parent config/voice_personalities.py
AGENT_VOICES = {
    "router": {
        "voice": "en_GB-alan-medium",
        "speed": 0.9,
        "pitch": 1.0
    },
    "office": {
        "voice": "en_US-amy-medium",
        "speed": 1.0,
        "pitch": 1.1
    }
}
```

### Audio Device Selection

List available audio devices:
```python
import sounddevice as sd
print(sd.query_devices())
```

Then set in config:
```toml
[system]
input_device_index = 2  # Your microphone
output_device_index = 3  # Your speakers
```

## Troubleshooting

### No Audio Input
- Check microphone permissions
- Verify device index in configuration
- Test with: `python -m sounddevice`

### Wake Word Not Detected
- Speak clearly and directly
- Adjust microphone position
- Try alternative wake phrases
- Check audio levels

### Poor Recognition Accuracy
- Use larger Whisper model
- Reduce background noise
- Speak more clearly
- Adjust VAD threshold

### TTS Voice Issues
- Ensure voice model files are downloaded
- Check file paths in configuration
- Verify .onnx.json config file exists

### WSL2 Audio
For WSL2 users, install PulseAudio:
```bash
# Windows side
Install VcXsrv or PulseAudio for Windows

# WSL2 side
sudo apt-get install pulseaudio
export PULSE_SERVER=tcp:$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
```

## Development

### Project Structure
```
voice-automation/
├── voice_router.py       # Main voice processing logic
├── voice_config.toml     # Configuration file
├── voices/              # TTS voice model directory
└── README.md           # This file
```

### Adding New Features

1. **Multi-Agent Wake Words**: Modify wake word detection to support multiple phrases
2. **Conversation Mode**: Add continuous listening without wake word
3. **Voice Commands**: Implement local command processing
4. **Audio Feedback**: Add more audio cues and confirmations

### Testing

Test components individually:
```python
# Test wake word detection
python -c "from voice_router import VoiceRouter; vr = VoiceRouter(); vr.test_wake_word()"

# Test ASR
python -c "from voice_router import VoiceRouter; vr = VoiceRouter(); vr.test_transcription()"

# Test TTS
python -c "from voice_router import VoiceRouter; vr = VoiceRouter(); vr.speak('Hello, world!')"
```

## Integration with MCP

The voice router sends transcribed text to the MCP endpoint as:
```json
{
  "prompt": "transcribed text here",
  "context": {
    "source": "voice",
    "wake_word": "deep thought",
    "timestamp": "2025-01-27T10:30:00Z"
  }
}
```

Responses are received via SSE and converted to speech.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test with different voices and models
4. Submit a pull request

## License

MIT License - See LICENSE file in parent repository

## Acknowledgments

- OpenWakeWord for wake word detection
- Faster-whisper for efficient ASR
- Piper TTS for high-quality voice synthesis
- Silero VAD for voice activity detection

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/hvkshetry/voice-automation/issues)
- Check parent project: [Codex MCP Orchestra](https://github.com/hvkshetry/codex-mcp-orchestra)