# FreeSWITCH Integration Methods

## Overview

FreeSWITCH provides multiple integration points for voicemail detection systems, from traditional module-based approaches to modern real-time audio streaming and AI service integration. This document covers all available methods, their implementation details, and best practices.

## Event Socket Library (ESL) Integration

### Python ESL Bindings

The Event Socket Library provides a programmatic interface to FreeSWITCH, enabling real-time call control and event handling for voicemail detection systems.

#### Basic ESL Connection

```python
import ESL
import json
import time

class FreeSWITCHVoicemailHandler:
    def __init__(self, host='localhost', port=8021, password='ClueCon'):
        self.host = host
        self.port = port
        self.password = password
        self.connection = None
    
    def connect(self):
        """Establish ESL connection to FreeSWITCH"""
        self.connection = ESL.ESLconnection(self.host, self.port, self.password)
        if self.connection.connected():
            print("Connected to FreeSWITCH via ESL")
            # Subscribe to events
            self.connection.events("plain", "CHANNEL_ANSWER CHANNEL_HANGUP CUSTOM")
            return True
        return False
    
    def handle_events(self):
        """Main event handling loop"""
        while self.connection.connected():
            event = self.connection.recvEvent()
            if event:
                self.process_event(event)
    
    def process_event(self, event):
        """Process incoming FreeSWITCH events"""
        event_name = event.getHeader("Event-Name")
        
        if event_name == "CHANNEL_ANSWER":
            self.handle_channel_answer(event)
        elif event_name == "CHANNEL_HANGUP":
            self.handle_channel_hangup(event)
        elif event_name == "CUSTOM":
            subclass = event.getHeader("Event-Subclass")
            if subclass == "avmd::beep":
                self.handle_voicemail_detected(event)
```

#### Call Control and Voicemail Detection

```python
def start_voicemail_detection(self, uuid):
    """Start voicemail detection on a specific call"""
    
    # Start AVMD if available
    cmd = f"uuid_broadcast {uuid} avmd_start::inbound_channel=1"
    result = self.connection.api(cmd)
    
    # Set up call recording for AI analysis
    recording_path = f"/tmp/vm_detection_{uuid}.wav"
    cmd = f"uuid_record {uuid} start {recording_path}"
    self.connection.api(cmd)
    
    # Set channel variables for tracking
    self.connection.api(f"uuid_setvar {uuid} vm_detection_active true")
    self.connection.api(f"uuid_setvar {uuid} vm_detection_start_time {int(time.time())}")
    
    return recording_path

def handle_voicemail_detected(self, event):
    """Handle voicemail detection event"""
    uuid = event.getHeader("Unique-ID")
    detection_method = event.getHeader("Event-Subclass")
    
    print(f"Voicemail detected on call {uuid} via {detection_method}")
    
    # Stop call recording
    self.connection.api(f"uuid_record {uuid} stop")
    
    # Transfer to voicemail handler
    self.connection.api(f"uuid_transfer {uuid} voicemail_redirect XML default")
    
    # Log detection event
    self.log_voicemail_detection(uuid, detection_method)
```

### ESL Event Handling for AI Integration

```python
class AIVoicemailDetectionHandler:
    def __init__(self):
        self.ai_detector = VoicemailAIDetector()
        self.active_detections = {}
    
    def start_ai_detection(self, uuid):
        """Start AI-based voicemail detection"""
        
        # Create WebSocket connection for audio streaming
        ws_url = f"ws://localhost:8080/audio_stream/{uuid}"
        
        # Store detection context
        self.active_detections[uuid] = {
            'start_time': time.time(),
            'audio_buffer': [],
            'ws_connection': None
        }
        
        # Configure FreeSWITCH to stream audio
        self.setup_audio_streaming(uuid, ws_url)
    
    def setup_audio_streaming(self, uuid, ws_url):
        """Configure FreeSWITCH to stream audio to AI service"""
        
        # Use mod_vg_tap_ws for real-time audio streaming
        cmd = f"uuid_broadcast {uuid} vg_tap_ws::start::{ws_url}"
        result = self.connection.api(cmd)
        
        if result.getBody() == "+OK":
            print(f"Audio streaming started for call {uuid}")
        else:
            print(f"Failed to start audio streaming: {result.getBody()}")
```

## WebRTC Integration with mod_verto

### Verto Configuration

FreeSWITCH's mod_verto provides WebRTC capabilities that can be leveraged for AI voicemail detection.

```xml
<!-- autoload_configs/verto.conf.xml -->
<configuration name="verto.conf" description="Verto Endpoint">
  <settings>
    <param name="bind-local" value="0.0.0.0:8081"/>
    <param name="bind-secure" value="0.0.0.0:8082"/>
    <param name="secure-cert" value="$${certs_dir}/wss.pem"/>
    <param name="secure-key" value="$${certs_dir}/wss.pem"/>
    <param name="secure-chain" value="$${certs_dir}/wss.pem"/>
  </settings>
  
  <profiles>
    <profile name="default">
      <param name="bind-local" value="0.0.0.0:8081"/>
      <param name="secure-bind" value="0.0.0.0:8082"/>
      <param name="userauth" value="true"/>
      <param name="context" value="public"/>
      <param name="dialplan" value="XML"/>
      <param name="auth-domain" value="$${domain}"/>
    </profile>
  </profiles>
</configuration>
```

### JavaScript WebRTC Client for AI Integration

```javascript
// WebRTC client for real-time voicemail detection
class VoicemailDetectionClient {
    constructor() {
        this.verto = null;
        this.currentSession = null;
        this.aiWebSocket = null;
    }
    
    connect() {
        const vertoCallbacks = {
            onWSConnect: this.onWSConnect.bind(this),
            onWSDisconnect: this.onWSDisconnect.bind(this),
            onDialogState: this.onDialogState.bind(this)
        };
        
        this.verto = new jQuery.verto({
            login: 'user@domain.com',
            passwd: 'password',
            socketUrl: 'wss://freeswitch.example.com:8082',
            callbacks: vertoCallbacks
        });
    }
    
    startCall(destination) {
        const callParams = {
            to: destination,
            from: 'ai-detector@domain.com',
            useVideo: false,
            useStereo: true,
            callbacks: {
                onStream: this.onStream.bind(this),
                onDialogState: this.onDialogState.bind(this)
            }
        };
        
        this.currentSession = this.verto.newCall(callParams);
    }
    
    onStream(session, stream) {
        // Stream audio to AI detection service
        this.connectToAIService();
        this.streamAudioToAI(stream);
    }
    
    connectToAIService() {
        this.aiWebSocket = new WebSocket('ws://ai-service.example.com:8080/detect');
        
        this.aiWebSocket.onmessage = (event) => {
            const result = JSON.parse(event.data);
            if (result.voicemail_detected) {
                this.handleVoicemailDetected(result);
            }
        };
    }
    
    streamAudioToAI(stream) {
        const audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        processor.onaudioprocess = (event) => {
            const audioData = event.inputBuffer.getChannelData(0);
            
            // Convert to format expected by AI service
            const int16Array = new Int16Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                int16Array[i] = audioData[i] * 32767;
            }
            
            // Send to AI service
            if (this.aiWebSocket && this.aiWebSocket.readyState === WebSocket.OPEN) {
                this.aiWebSocket.send(int16Array.buffer);
            }
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
    }
}
```

## Real-Time Audio Tapping with mod_vg_tap_ws

### Configuration

```xml
<!-- Load mod_vg_tap_ws for audio streaming -->
<!-- modules.conf.xml -->
<load module="mod_vg_tap_ws"/>
```

### Audio Streaming Setup

```lua
-- Lua script to start audio streaming for AI analysis
-- /etc/freeswitch/scripts/start_ai_detection.lua

function start_ai_voicemail_detection()
    local uuid = session:get_uuid()
    local ai_service_url = "ws://ai-detector.local:8080/stream/" .. uuid
    
    -- Start audio streaming to AI service
    session:execute("vg_tap_ws", "start " .. ai_service_url .. " both")
    
    -- Set channel variables
    session:setVariable("ai_detection_active", "true")
    session:setVariable("ai_detection_url", ai_service_url)
    
    freeswitch.consoleLog("INFO", "AI voicemail detection started for call " .. uuid)
end

-- Execute the function
start_ai_voicemail_detection()
```

### Python WebSocket Server for AI Processing

```python
import asyncio
import websockets
import json
import numpy as np
from voicemail_ai_detector import VoicemailDetector

class AudioStreamProcessor:
    def __init__(self):
        self.detector = VoicemailDetector()
        self.active_streams = {}
    
    async def handle_audio_stream(self, websocket, path):
        """Handle incoming audio stream from FreeSWITCH"""
        
        # Extract call UUID from path
        call_uuid = path.split('/')[-1]
        
        print(f"Audio stream started for call {call_uuid}")
        
        # Initialize stream context
        self.active_streams[call_uuid] = {
            'websocket': websocket,
            'audio_buffer': np.array([]),
            'detection_started': False,
            'start_time': asyncio.get_event_loop().time()
        }
        
        try:
            async for message in websocket:
                await self.process_audio_chunk(call_uuid, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"Audio stream ended for call {call_uuid}")
        finally:
            if call_uuid in self.active_streams:
                del self.active_streams[call_uuid]
    
    async def process_audio_chunk(self, call_uuid, audio_data):
        """Process incoming audio chunk"""
        
        stream_info = self.active_streams[call_uuid]
        
        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
        
        # Add to buffer
        stream_info['audio_buffer'] = np.concatenate([
            stream_info['audio_buffer'], 
            audio_chunk
        ])
        
        # Analyze when we have enough audio (2 seconds at 8kHz)
        if len(stream_info['audio_buffer']) >= 16000 and not stream_info['detection_started']:
            stream_info['detection_started'] = True
            
            # Run AI detection
            is_voicemail = await self.detect_voicemail(
                stream_info['audio_buffer'][:16000]
            )
            
            if is_voicemail:
                await self.send_detection_result(call_uuid, True)
    
    async def detect_voicemail(self, audio_data):
        """Run AI voicemail detection on audio data"""
        
        # Run inference (this should be async to avoid blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.detector.detect, 
            audio_data
        )
        
        return result
    
    async def send_detection_result(self, call_uuid, is_voicemail):
        """Send detection result back to FreeSWITCH"""
        
        # Send result via websocket
        result = {
            'call_uuid': call_uuid,
            'voicemail_detected': is_voicemail,
            'confidence': 0.95,
            'detection_time': asyncio.get_event_loop().time()
        }
        
        stream_info = self.active_streams[call_uuid]
        await stream_info['websocket'].send(json.dumps(result))
        
        # Also send to FreeSWITCH via ESL
        await self.notify_freeswitch(call_uuid, result)
    
    async def notify_freeswitch(self, call_uuid, result):
        """Notify FreeSWITCH of detection result via ESL"""
        
        import ESL
        
        # Connect to FreeSWITCH ESL
        con = ESL.ESLconnection('localhost', 8021, 'ClueCon')
        
        if con.connected():
            if result['voicemail_detected']:
                # Set channel variable and transfer
                con.api(f"uuid_setvar {call_uuid} ai_voicemail_detected true")
                con.api(f"uuid_transfer {call_uuid} voicemail_handler XML default")
            
            con.disconnect()

# Start WebSocket server
async def main():
    processor = AudioStreamProcessor()
    
    server = await websockets.serve(
        processor.handle_audio_stream,
        "localhost",
        8080
    )
    
    print("AI audio processing server started on localhost:8080")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
```

## MRCP Integration for Cloud AI Services

### MRCP Configuration

```xml
<!-- autoload_configs/unimrcp.conf.xml -->
<configuration name="unimrcp.conf" description="UniMRCP Client">
  <settings>
    <param name="default-tts-profile" value="speechsynth"/>
    <param name="default-asr-profile" value="speechrecog"/>
    <param name="log-level" value="DEBUG"/>
    <param name="max-connection-count" value="100"/>
    <param name="offer-new-connection" value="1"/>
  </settings>
  
  <profiles>
    <profile name="speechrecog">
      <param name="version" value="2"/>
      <param name="server-ip" value="ai-service.example.com"/>
      <param name="server-port" value="1544"/>
      <param name="resource-location" value=""/>
      <param name="speechrecog" value="recognizer"/>
      <param name="rtp-ip" value="auto"/>
      <param name="rtp-port-min" value="4000"/>
      <param name="rtp-port-max" value="5000"/>
      <param name="codecs" value="PCMU PCMA L16/96/8000"/>
    </profile>
  </profiles>
</configuration>
```

### Dialplan Integration with MRCP

```xml
<!-- Dialplan extension for MRCP-based AI detection -->
<extension name="ai_voicemail_detection">
  <condition field="destination_number" expression="^(.*)$">
    <action application="answer"/>
    <action application="set" data="tts_engine=unimrcp"/>
    <action application="set" data="tts_voice=speechsynth"/>
    
    <!-- Start bridge with AI monitoring -->
    <action application="set" data="continue_on_fail=true"/>
    <action application="set" data="hangup_after_bridge=false"/>
    
    <!-- Use MRCP for speech recognition during bridge -->
    <action application="play_and_detect_speech" 
            data="silence_stream://1000 detect:unimrcp {start-input-timers=false,no-input-timeout=5000,recognition-timeout=10000}builtin:speech/builtin grammars/boolean.grxml"/>
    
    <!-- Bridge the call -->
    <action application="bridge" data="sofia/gateway/${routing_provider}/${routing_did}"/>
    
    <!-- Check if voicemail was detected -->
    <action application="lua" data="handle_ai_detection_result.lua"/>
  </condition>
</extension>
```

## Custom Module Development

### Creating a Custom Voicemail Detection Module

```c
// mod_ai_voicemail_detector.c
#include <switch.h>

SWITCH_MODULE_LOAD_FUNCTION(mod_ai_voicemail_detector_load);
SWITCH_MODULE_SHUTDOWN_FUNCTION(mod_ai_voicemail_detector_shutdown);
SWITCH_MODULE_DEFINITION(mod_ai_voicemail_detector, 
                        mod_ai_voicemail_detector_load, 
                        mod_ai_voicemail_detector_shutdown, 
                        NULL);

typedef struct {
    switch_core_session_t *session;
    switch_channel_t *channel;
    switch_memory_pool_t *pool;
    char *ai_service_url;
    int detection_active;
    switch_mutex_t *mutex;
} ai_detector_context_t;

static switch_status_t ai_detector_init(switch_core_session_t *session, 
                                       const char *data, 
                                       ai_detector_context_t **context) {
    
    ai_detector_context_t *ctx;
    switch_memory_pool_t *pool = switch_core_session_get_pool(session);
    
    ctx = switch_core_alloc(pool, sizeof(ai_detector_context_t));
    ctx->session = session;
    ctx->channel = switch_core_session_get_channel(session);
    ctx->pool = pool;
    ctx->ai_service_url = switch_core_strdup(pool, data);
    ctx->detection_active = 1;
    
    switch_mutex_init(&ctx->mutex, SWITCH_MUTEX_NESTED, pool);
    
    *context = ctx;
    return SWITCH_STATUS_SUCCESS;
}

static switch_bool_t ai_detector_audio_callback(switch_media_bug_t *bug, 
                                               void *user_data, 
                                               switch_abc_type_t type) {
    
    ai_detector_context_t *ctx = (ai_detector_context_t *)user_data;
    switch_frame_t *frame = NULL;
    
    switch (type) {
    case SWITCH_ABC_TYPE_INIT:
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, 
                         "AI Voicemail Detector initialized\n");
        break;
        
    case SWITCH_ABC_TYPE_READ_REPLACE:
    case SWITCH_ABC_TYPE_READ:
        if ((frame = switch_core_media_bug_get_read_replace_frame(bug))) {
            // Send audio data to AI service
            send_audio_to_ai_service(ctx, frame);
        }
        break;
        
    case SWITCH_ABC_TYPE_CLOSE:
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, 
                         "AI Voicemail Detector closed\n");
        break;
        
    default:
        break;
    }
    
    return SWITCH_TRUE;
}

SWITCH_STANDARD_APP(ai_voicemail_detector_start) {
    switch_media_bug_t *bug;
    ai_detector_context_t *ctx = NULL;
    
    if (ai_detector_init(session, data, &ctx) != SWITCH_STATUS_SUCCESS) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), 
                         SWITCH_LOG_ERROR, 
                         "Failed to initialize AI detector\n");
        return;
    }
    
    if (switch_core_media_bug_add(session, 
                                 "ai_voicemail_detector", 
                                 NULL,
                                 ai_detector_audio_callback, 
                                 ctx, 
                                 0, 
                                 SMBF_READ_STREAM | SMBF_NO_PAUSE,
                                 &bug) != SWITCH_STATUS_SUCCESS) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), 
                         SWITCH_LOG_ERROR, 
                         "Failed to attach AI detector bug\n");
        return;
    }
    
    switch_channel_set_private(switch_core_session_get_channel(session), 
                              "ai_detector_bug", bug);
}

SWITCH_MODULE_LOAD_FUNCTION(mod_ai_voicemail_detector_load) {
    switch_application_interface_t *app_interface;
    
    *module_interface = switch_loadable_module_create_module_interface(pool, 
                                                                      modname);
    
    SWITCH_ADD_APP(app_interface, 
                   "ai_voicemail_detector_start", 
                   "Start AI voicemail detection", 
                   "Start AI voicemail detection", 
                   ai_voicemail_detector_start, 
                   "<ai_service_url>", 
                   SAF_NONE);
    
    return SWITCH_STATUS_SUCCESS;
}
```

## Integration Performance Considerations

### Latency Optimization

```python
# Optimized audio processing for low latency
class LowLatencyAudioProcessor:
    def __init__(self, target_latency_ms=100):
        self.target_latency = target_latency_ms / 1000.0
        self.buffer_size = int(8000 * self.target_latency)  # 8kHz sample rate
        self.detector = OptimizedVoicemailDetector()
    
    async def process_realtime_audio(self, audio_stream):
        """Process audio with minimal latency"""
        
        buffer = np.array([])
        
        async for chunk in audio_stream:
            buffer = np.concatenate([buffer, chunk])
            
            # Process when buffer reaches target size
            if len(buffer) >= self.buffer_size:
                # Use only the latest audio for detection
                analysis_chunk = buffer[-self.buffer_size:]
                
                # Quick detection with optimized model
                start_time = time.time()
                result = await self.detector.fast_detect(analysis_chunk)
                processing_time = time.time() - start_time
                
                if processing_time > self.target_latency:
                    print(f"Warning: Processing time {processing_time:.3f}s exceeds target {self.target_latency:.3f}s")
                
                if result['is_voicemail']:
                    return result
                
                # Keep small buffer for continuity
                buffer = buffer[-int(self.buffer_size * 0.1):]
        
        return {'is_voicemail': False, 'confidence': 0.0}
```

### Memory Management

```python
class MemoryEfficientDetector:
    def __init__(self, max_concurrent_calls=100):
        self.max_concurrent = max_concurrent_calls
        self.active_calls = {}
        self.memory_pool = {}
    
    def allocate_call_resources(self, call_uuid):
        """Allocate memory resources for call processing"""
        
        if len(self.active_calls) >= self.max_concurrent:
            # Clean up oldest inactive calls
            self.cleanup_inactive_calls()
        
        # Reuse memory pool if available
        if self.memory_pool:
            resources = self.memory_pool.pop()
        else:
            resources = {
                'audio_buffer': np.zeros(32000, dtype=np.float32),
                'feature_buffer': np.zeros(512, dtype=np.float32),
                'temp_arrays': [np.zeros(1024) for _ in range(4)]
            }
        
        self.active_calls[call_uuid] = {
            'resources': resources,
            'last_activity': time.time()
        }
        
        return resources
    
    def release_call_resources(self, call_uuid):
        """Release memory resources back to pool"""
        
        if call_uuid in self.active_calls:
            resources = self.active_calls[call_uuid]['resources']
            
            # Clear arrays but keep allocated memory
            for key, array in resources.items():
                if isinstance(array, np.ndarray):
                    array.fill(0)
                elif isinstance(array, list):
                    for arr in array:
                        if isinstance(arr, np.ndarray):
                            arr.fill(0)
            
            # Return to pool for reuse
            self.memory_pool[call_uuid] = resources
            del self.active_calls[call_uuid]
```

## Error Handling and Resilience

### Connection Failure Handling

```python
class ResilientVoicemailDetector:
    def __init__(self):
        self.primary_ai_service = "ws://ai-primary.local:8080"
        self.backup_ai_service = "ws://ai-backup.local:8080"
        self.fallback_detector = TraditionalAVMDDetector()
        self.connection_pool = []
    
    async def detect_with_fallback(self, audio_data, call_uuid):
        """Detect voicemail with multiple fallback options"""
        
        # Try primary AI service
        try:
            result = await self.try_ai_detection(
                self.primary_ai_service, audio_data
            )
            if result:
                return result
        except Exception as e:
            print(f"Primary AI service failed: {e}")
        
        # Try backup AI service
        try:
            result = await self.try_ai_detection(
                self.backup_ai_service, audio_data
            )
            if result:
                return result
        except Exception as e:
            print(f"Backup AI service failed: {e}")
        
        # Fall back to traditional detection
        try:
            result = self.fallback_detector.detect(audio_data)
            result['method'] = 'fallback_avmd'
            return result
        except Exception as e:
            print(f"Fallback detection failed: {e}")
            return {'is_voicemail': False, 'confidence': 0.0, 'method': 'none'}
    
    async def try_ai_detection(self, service_url, audio_data, timeout=2.0):
        """Try AI detection with timeout"""
        
        try:
            async with websockets.connect(service_url) as websocket:
                # Send audio data
                await websocket.send(audio_data.tobytes())
                
                # Wait for result with timeout
                result = await asyncio.wait_for(
                    websocket.recv(), timeout=timeout
                )
                
                return json.loads(result)
                
        except asyncio.TimeoutError:
            print(f"AI service timeout: {service_url}")
            return None
        except Exception as e:
            print(f"AI service error: {e}")
            return None
```

## Monitoring and Metrics

### Detection Performance Monitoring

```python
class VoicemailDetectionMonitor:
    def __init__(self):
        self.metrics = {
            'total_calls': 0,
            'voicemail_detected': 0,
            'human_detected': 0,
            'detection_times': [],
            'accuracy_feedback': {'correct': 0, 'incorrect': 0},
            'method_usage': {'ai': 0, 'avmd': 0, 'fallback': 0}
        }
    
    def record_detection(self, result, processing_time, method='ai'):
        """Record detection result and performance metrics"""
        
        self.metrics['total_calls'] += 1
        self.metrics['detection_times'].append(processing_time)
        self.metrics['method_usage'][method] += 1
        
        if result['is_voicemail']:
            self.metrics['voicemail_detected'] += 1
        else:
            self.metrics['human_detected'] += 1
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        
        if not self.metrics['detection_times']:
            return {}
        
        times = self.metrics['detection_times']
        
        return {
            'total_calls': self.metrics['total_calls'],
            'voicemail_rate': self.metrics['voicemail_detected'] / self.metrics['total_calls'],
            'avg_detection_time': np.mean(times),
            'max_detection_time': max(times),
            'min_detection_time': min(times),
            'method_distribution': self.metrics['method_usage'],
            'accuracy_rate': self.calculate_accuracy_rate()
        }
    
    def export_metrics_to_prometheus(self):
        """Export metrics in Prometheus format"""
        
        stats = self.get_performance_stats()
        
        prometheus_metrics = f"""
# HELP voicemail_detection_total Total number of voicemail detections
# TYPE voicemail_detection_total counter
voicemail_detection_total {self.metrics['total_calls']}

# HELP voicemail_detection_rate Rate of voicemail vs human detection
# TYPE voicemail_detection_rate gauge
voicemail_detection_rate {stats.get('voicemail_rate', 0)}

# HELP voicemail_detection_duration_seconds Time taken for detection
# TYPE voicemail_detection_duration_seconds histogram
voicemail_detection_duration_seconds_sum {sum(self.metrics['detection_times'])}
voicemail_detection_duration_seconds_count {len(self.metrics['detection_times'])}
"""
        
        return prometheus_metrics
```

## Conclusion

FreeSWITCH provides multiple robust integration points for voicemail detection systems:

1. **Event Socket Library (ESL)**: Real-time call control and event handling
2. **WebRTC with mod_verto**: Browser-based integration with AI services
3. **Audio Streaming**: Real-time audio tapping with mod_vg_tap_ws
4. **MRCP Integration**: Cloud-based AI service connectivity
5. **Custom Modules**: Native C modules for high-performance processing

Key success factors for FreeSWITCH integration:

- **Choose the right integration method** based on latency and scalability requirements
- **Implement proper error handling** and fallback mechanisms
- **Monitor performance metrics** continuously
- **Optimize memory usage** for high-volume scenarios
- **Use connection pooling** for external AI services

The combination of FreeSWITCH's flexible architecture and modern AI services enables the development of highly accurate and scalable voicemail detection systems suitable for production telephony environments.