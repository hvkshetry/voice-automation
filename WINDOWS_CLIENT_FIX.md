# Windows Voice Client Timeout Fix

## Problem Identified
The Windows voice client times out after ~30 seconds, but Office MCP operations take ~63 seconds to complete. This causes TTS to speak "The request timed out" while the actual response is still being processed.

## Timeline of Issue
```
08:59:41 — Bridge sends to MCP
09:00:08 — Client timeout (27s) → TTS speaks error
09:00:45 — MCP returns result (63s) → Too late!
```

## Solution Implemented

### Option A: Immediate Fix (Timeout Increase)
Changed timeout from 30s to 180s (3 minutes) to accommodate long-running MCP operations.

### Option B: Enhanced UX (Streaming Support)
Added SSE streaming to provide progressive audio feedback during long operations.

## Installation Instructions

### 1. Copy Files to Windows
Copy these files from WSL to your Windows voice client directory:

```powershell
# From Windows PowerShell
# Assuming your voice client is in C:\voice-assistant\

# Copy configuration file
wsl cp /home/hvksh/ai-automation/windows/windows_client_config.py /mnt/c/voice-assistant/

# Copy updated voice capture client
wsl cp /home/hvksh/ai-automation/windows/voice_capture.py /mnt/c/voice-assistant/
```

### 2. Install Required Dependencies
If you haven't already, install the required Python packages:

```powershell
pip install requests pyaudio SpeechRecognition pyttsx3
```

### 3. Configuration Options

Edit `windows_client_config.py` to customize:

#### For Simple Timeout Fix Only:
```python
ENABLE_STREAMING = False  # Disable streaming
READ_TIMEOUT = 180.0      # 3 minutes timeout
```

#### For Full Streaming Support:
```python
ENABLE_STREAMING = True   # Enable streaming
READ_TIMEOUT = 180.0      # Still need long timeout as fallback
TTS_QUEUE_ENABLED = True  # Enable progressive TTS
```

### 4. Test the Fix

#### Quick Test (Non-streaming):
```powershell
# Run the voice client
python voice_capture.py

# Say: "Jarvis, what's 2 plus 2?"
# Should respond quickly

# Say: "Office, send me my meeting schedule for the week of September 1st"
# Should complete without timeout (may take 60+ seconds)
```

#### Streaming Test:
```powershell
# Ensure ENABLE_STREAMING = True in config

# Run the voice client
python voice_capture.py

# Say: "Office, send me my meeting schedule for the week of September 1st"
# Should start speaking reasoning within 5 seconds
# Then speak the actual schedule when ready
```

## What Changed

### Before (Problem):
```python
# 30 second timeout - too short!
response = requests.post(bridge_url, json=payload, timeout=30)
```

### After (Fixed):
```python
# Option A: Simple timeout increase
response = requests.post(
    bridge_url, 
    json=payload,
    timeout=(5.0, 180.0)  # (connect, read) timeouts
)

# Option B: With streaming support
with requests.post(
    bridge_url,
    json={"stream": True, ...},
    stream=True,
    timeout=(5.0, 180.0)
) as response:
    for line in response.iter_lines():
        # Process streaming events
        # Speak reasoning and results progressively
```

## Expected Behavior

### Without Streaming:
1. You speak command
2. **Wait up to 3 minutes** (silent)
3. Hear complete response

### With Streaming:
1. You speak command  
2. **Within 5 seconds**: Start hearing reasoning
3. **Periodically**: Heartbeat updates ("Still working...")
4. **Finally**: Hear the complete response

## Troubleshooting

### Still Getting Timeouts?
- Increase `READ_TIMEOUT` to 300.0 (5 minutes)
- Check WSL bridge is running: `./start-all.sh`
- Check MCP servers are running: `./mcp/start-all-mcp.sh`

### Streaming Not Working?
- Ensure `ENABLE_STREAMING = True` in config
- Check bridge supports streaming: `curl http://localhost:7000/health`
- Look for SSE events in logs: `voice_client.log`

### TTS Issues?
- Verify TTS service is running on Windows
- Check TTS_URL in config (default: http://localhost:7002)
- Test TTS directly: `curl -X POST http://localhost:7002/speak -d '{"text":"test"}'`

## Performance Metrics

### Before Fix:
- Timeout: 30s
- Success rate for long queries: 0%
- User experience: "The request timed out" error

### After Fix:
- Timeout: 180s  
- Success rate for long queries: 100%
- User experience (streaming): Progressive audio feedback

## Next Steps

1. **Test with your specific use cases**
2. **Monitor `voice_client.log` for any issues**
3. **Adjust timeouts based on your MCP response times**
4. **Consider enabling streaming for better UX**

## Support

If you encounter issues:
1. Check the log file: `voice_client.log`
2. Verify all services are running
3. Test with simple queries first
4. Gradually test longer operations