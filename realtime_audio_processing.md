# Real-Time Audio Processing for Voicemail Detection

## Overview

Real-time audio processing is critical for immediate voicemail detection in telephony systems. This document covers the technologies, frameworks, and implementation strategies for processing live audio streams with minimal latency while maintaining high detection accuracy.

## Core Technologies

### Python WebRTC Implementations

#### aiortc Library

The `aiortc` library provides a comprehensive Python implementation of WebRTC, enabling real-time audio streaming and processing.

```python
import asyncio
import logging
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.mediastreams import MediaStreamTrack

class VoicemailDetectionTrack(MediaStreamTrack):
    """Custom audio track for voicemail detection"""
    
    kind = "audio"
    
    def __init__(self, track, detector):
        super().__init__()
        self.track = track
        self.detector = detector
        self.audio_buffer = np.array([])
        self.detection_threshold = 2.0  # seconds
        self.sample_rate = 8000
        
    async def recv(self):
        """Receive and process audio frames"""
        
        frame = await self.track.recv()
        
        # Convert audio frame to numpy array
        audio_data = np.frombuffer(
            frame.to_ndarray(), 
            dtype=np.float32
        ).flatten()
        
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        
        # Process when we have enough audio
        target_samples = int(self.detection_threshold * self.sample_rate)
        if len(self.audio_buffer) >= target_samples:
            # Run detection on buffered audio
            await self.process_detection()
            
            # Keep only recent audio for continuous analysis
            keep_samples = int(0.5 * self.sample_rate)
            self.audio_buffer = self.audio_buffer[-keep_samples:]
        
        return frame
    
    async def process_detection(self):
        """Process audio buffer for voicemail detection"""
        
        detection_result = await self.detector.detect_async(
            self.audio_buffer[:int(self.detection_threshold * self.sample_rate)]
        )
        
        if detection_result['is_voicemail']:
            # Emit detection event
            await self.emit_voicemail_event(detection_result)
    
    async def emit_voicemail_event(self, result):
        """Emit voicemail detection event"""
        
        event = {
            'type': 'voicemail_detected',
            'confidence': result['confidence'],
            'method': result.get('method', 'ai'),
            'timestamp': time.time()
        }
        
        # Custom event handling logic here
        print(f"Voicemail detected: {event}")

# WebRTC Peer Connection Setup
async def create_voicemail_detection_peer():
    """Create WebRTC peer connection with voicemail detection"""
    
    pc = RTCPeerConnection()
    detector = VoicemailDetector()
    
    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            # Wrap incoming audio track with detection capability
            detection_track = VoicemailDetectionTrack(track, detector)
            
            # Add processed track back to peer connection
            pc.addTrack(detection_track)
    
    return pc
```

#### Real-Time Audio Streaming Server

```python
import websockets
import json
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import object_from_string, object_to_string

class VoicemailDetectionServer:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.active_connections = {}
        self.detector = VoicemailAIDetector()
    
    async def handle_signaling(self, websocket, path):
        """Handle WebRTC signaling and audio processing"""
        
        connection_id = id(websocket)
        pc = RTCPeerConnection()
        
        self.active_connections[connection_id] = {
            'pc': pc,
            'websocket': websocket,
            'audio_processor': None
        }
        
        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                # Create real-time audio processor
                processor = RealTimeAudioProcessor(
                    track, self.detector, websocket
                )
                self.active_connections[connection_id]['audio_processor'] = processor
                
                # Start processing
                asyncio.create_task(processor.start_processing())
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'offer':
                    # Handle WebRTC offer
                    offer = object_from_string(data['sdp'])
                    await pc.setRemoteDescription(offer)
                    
                    # Create answer
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    
                    # Send answer back
                    await websocket.send(json.dumps({
                        'type': 'answer',
                        'sdp': object_to_string(answer)
                    }))
                
                elif data['type'] == 'ice-candidate':
                    # Handle ICE candidate
                    candidate = object_from_string(data['candidate'])
                    await pc.addIceCandidate(candidate)
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await pc.close()
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
    
    async def start_server(self):
        """Start the WebRTC signaling server"""
        
        server = await websockets.serve(
            self.handle_signaling,
            self.host,
            self.port
        )
        
        print(f"Voicemail detection server started on {self.host}:{self.port}")
        await server.wait_closed()

class RealTimeAudioProcessor:
    def __init__(self, audio_track, detector, websocket):
        self.audio_track = audio_track
        self.detector = detector
        self.websocket = websocket
        self.is_processing = False
        self.audio_buffer = np.array([])
        self.sample_rate = 8000
    
    async def start_processing(self):
        """Start real-time audio processing"""
        
        self.is_processing = True
        
        while self.is_processing:
            try:
                # Receive audio frame
                frame = await asyncio.wait_for(
                    self.audio_track.recv(), 
                    timeout=1.0
                )
                
                # Process frame
                await self.process_frame(frame)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
                break
    
    async def process_frame(self, frame):
        """Process individual audio frame"""
        
        # Convert frame to numpy array
        audio_data = frame.to_ndarray().astype(np.float32).flatten()
        
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        
        # Process when buffer reaches 2 seconds
        target_length = 2 * self.sample_rate
        if len(self.audio_buffer) >= target_length:
            # Extract analysis window
            analysis_audio = self.audio_buffer[:target_length]
            
            # Run detection
            result = await self.detector.detect_async(analysis_audio)
            
            if result['is_voicemail']:
                # Send detection result
                await self.websocket.send(json.dumps({
                    'type': 'voicemail_detected',
                    'confidence': result['confidence'],
                    'processing_time': result.get('processing_time', 0)
                }))
                
                # Stop processing after detection
                self.is_processing = False
            
            # Slide buffer window
            self.audio_buffer = self.audio_buffer[target_length // 2:]
```

### TensorFlow and PyTorch Integration

#### TensorFlow Audio Processing

```python
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import load_model

class TensorFlowAudioProcessor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.sample_rate = 8000
        self.frame_length = 1024
        self.frame_step = 512
    
    @tf.function
    def preprocess_audio(self, audio_tensor):
        """Preprocess audio for model inference"""
        
        # Ensure correct sample rate
        if tf.shape(audio_tensor)[0] != self.sample_rate * 2:  # 2 seconds
            audio_tensor = tf.image.resize(
                tf.expand_dims(audio_tensor, axis=-1),
                [self.sample_rate * 2, 1]
            )[:, 0]
        
        # Compute STFT
        stft = tf.signal.stft(
            audio_tensor,
            frame_length=self.frame_length,
            frame_step=self.frame_step
        )
        
        # Convert to magnitude spectrogram
        magnitude = tf.abs(stft)
        
        # Convert to mel-scale
        mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=128,
            num_spectrogram_bins=self.frame_length // 2 + 1,
            sample_rate=self.sample_rate
        )
        
        mel_spec = tf.tensordot(magnitude, mel_spectrogram, 1)
        
        # Convert to log scale
        log_mel_spec = tf.math.log(mel_spec + 1e-6)
        
        return tf.expand_dims(log_mel_spec, axis=0)  # Add batch dimension
    
    async def detect_voicemail_async(self, audio_array):
        """Async voicemail detection using TensorFlow"""
        
        # Convert numpy array to TensorFlow tensor
        audio_tensor = tf.constant(audio_array, dtype=tf.float32)
        
        # Preprocess
        processed_audio = self.preprocess_audio(audio_tensor)
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            None, 
            self.model.predict, 
            processed_audio
        )
        
        confidence = float(prediction[0][0])
        is_voicemail = confidence > 0.5
        
        return {
            'is_voicemail': is_voicemail,
            'confidence': confidence,
            'method': 'tensorflow_cnn'
        }

# GPU-accelerated batch processing
class GPUAudioBatchProcessor:
    def __init__(self, model_path, batch_size=8):
        # Configure GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        self.model = load_model(model_path)
        self.batch_size = batch_size
        self.audio_queue = asyncio.Queue(maxsize=batch_size * 2)
        self.result_queues = {}
    
    async def process_audio_batch(self, audio_id, audio_data):
        """Add audio to processing batch"""
        
        # Create result queue for this audio
        result_queue = asyncio.Queue(maxsize=1)
        self.result_queues[audio_id] = result_queue
        
        # Add to processing queue
        await self.audio_queue.put((audio_id, audio_data))
        
        # Wait for result
        result = await result_queue.get()
        del self.result_queues[audio_id]
        
        return result
    
    async def batch_processing_worker(self):
        """Worker that processes audio in batches"""
        
        while True:
            batch_items = []
            
            # Collect batch
            for _ in range(self.batch_size):
                try:
                    item = await asyncio.wait_for(
                        self.audio_queue.get(), 
                        timeout=0.1
                    )
                    batch_items.append(item)
                except asyncio.TimeoutError:
                    break
            
            if not batch_items:
                await asyncio.sleep(0.01)
                continue
            
            # Process batch
            audio_ids = [item[0] for item in batch_items]
            audio_batch = np.array([item[1] for item in batch_items])
            
            # Preprocess batch
            processed_batch = []
            for audio in audio_batch:
                processed = self.preprocess_audio(
                    tf.constant(audio, dtype=tf.float32)
                )
                processed_batch.append(processed[0])  # Remove batch dim
            
            batch_tensor = tf.stack(processed_batch)
            
            # Run batch inference
            predictions = self.model.predict(batch_tensor)
            
            # Send results
            for audio_id, prediction in zip(audio_ids, predictions):
                result = {
                    'is_voicemail': float(prediction[0]) > 0.5,
                    'confidence': float(prediction[0]),
                    'method': 'tensorflow_batch'
                }
                
                if audio_id in self.result_queues:
                    await self.result_queues[audio_id].put(result)
```

#### PyTorch Real-Time Inference

```python
import torch
import torchaudio
from torch.utils.data import DataLoader
import asyncio
from concurrent.futures import ThreadPoolExecutor

class PyTorchVoicemailDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.sample_rate = 8000
        self.target_length = 2 * self.sample_rate  # 2 seconds
        
        # Thread pool for CPU preprocessing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def preprocess_audio(self, audio_numpy):
        """Preprocess audio for PyTorch model"""
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_numpy).float()
        
        # Ensure correct length
        if len(audio_tensor) > self.target_length:
            audio_tensor = audio_tensor[:self.target_length]
        elif len(audio_tensor) < self.target_length:
            padding = self.target_length - len(audio_tensor)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        
        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )
        
        mel_spec = mel_transform(audio_tensor)
        
        # Convert to log scale
        log_mel = torch.log(mel_spec + 1e-6)
        
        # Add batch and channel dimensions
        return log_mel.unsqueeze(0).unsqueeze(0)
    
    async def detect_voicemail_async(self, audio_array):
        """Async voicemail detection using PyTorch"""
        
        # Preprocess in thread pool
        loop = asyncio.get_event_loop()
        preprocessed = await loop.run_in_executor(
            self.executor,
            self.preprocess_audio,
            audio_array
        )
        
        # Move to device and run inference
        preprocessed = preprocessed.to(self.device)
        
        with torch.no_grad():
            # Run inference in thread pool to avoid blocking
            prediction = await loop.run_in_executor(
                self.executor,
                self._run_inference,
                preprocessed
            )
        
        confidence = float(torch.sigmoid(prediction).item())
        is_voicemail = confidence > 0.5
        
        return {
            'is_voicemail': is_voicemail,
            'confidence': confidence,
            'method': 'pytorch_cnn'
        }
    
    def _run_inference(self, input_tensor):
        """Run model inference (blocking)"""
        return self.model(input_tensor)

# Real-time streaming processor with PyTorch
class PyTorchStreamProcessor:
    def __init__(self, model_path, stream_buffer_size=4096):
        self.detector = PyTorchVoicemailDetector(model_path)
        self.buffer_size = stream_buffer_size
        self.audio_buffer = torch.zeros(0)
        self.sample_rate = 8000
        self.detection_window = 2 * self.sample_rate
    
    async def process_audio_stream(self, audio_stream):
        """Process continuous audio stream"""
        
        async for audio_chunk in audio_stream:
            # Convert chunk to tensor
            chunk_tensor = torch.from_numpy(audio_chunk).float()
            
            # Add to buffer
            self.audio_buffer = torch.cat([self.audio_buffer, chunk_tensor])
            
            # Process when buffer is large enough
            if len(self.audio_buffer) >= self.detection_window:
                # Extract analysis window
                analysis_window = self.audio_buffer[:self.detection_window]
                
                # Run detection
                result = await self.detector.detect_voicemail_async(
                    analysis_window.numpy()
                )
                
                if result['is_voicemail']:
                    return result
                
                # Slide window
                overlap = self.detection_window // 4
                self.audio_buffer = self.audio_buffer[self.detection_window - overlap:]
        
        return {'is_voicemail': False, 'confidence': 0.0}
```

### Voice Activity Detection (VAD)

#### WebRTC VAD Integration

```python
import webrtcvad
import collections
import numpy as np

class WebRTCVAD:
    def __init__(self, aggressiveness=3, sample_rate=8000, frame_duration=30):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        # Ring buffer for voice activity
        self.ring_buffer = collections.deque(maxlen=10)
        
    def is_speech(self, audio_frame):
        """Determine if audio frame contains speech"""
        
        # Ensure frame is correct size
        if len(audio_frame) != self.frame_size:
            return False
        
        # Convert to 16-bit PCM
        pcm_data = (audio_frame * 32767).astype(np.int16).tobytes()
        
        # Run VAD
        return self.vad.is_speech(pcm_data, self.sample_rate)
    
    def process_audio_stream(self, audio_stream):
        """Process audio stream with voice activity detection"""
        
        voiced_frames = []
        silence_frames = []
        
        # Process audio in frames
        for i in range(0, len(audio_stream) - self.frame_size, self.frame_size):
            frame = audio_stream[i:i + self.frame_size]
            
            is_voiced = self.is_speech(frame)
            self.ring_buffer.append((frame, is_voiced))
            
            if is_voiced:
                voiced_frames.append(frame)
            else:
                silence_frames.append(frame)
        
        return {
            'voiced_frames': np.concatenate(voiced_frames) if voiced_frames else np.array([]),
            'silence_frames': np.concatenate(silence_frames) if silence_frames else np.array([]),
            'voice_activity_ratio': len(voiced_frames) / max(1, len(voiced_frames) + len(silence_frames))
        }

# Advanced VAD with machine learning
class MLBasedVAD:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.frame_size = 1024
        self.hop_length = 512
    
    def extract_vad_features(self, audio_frame):
        """Extract features for VAD classification"""
        
        # Energy-based features
        energy = np.sum(audio_frame ** 2)
        zero_crossing_rate = np.sum(np.abs(np.diff(np.sign(audio_frame))))
        
        # Spectral features
        fft = np.fft.fft(audio_frame)
        magnitude = np.abs(fft[:len(fft)//2])
        spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)
        spectral_rolloff = np.where(np.cumsum(magnitude) >= 0.85 * np.sum(magnitude))[0][0]
        
        return np.array([
            energy,
            zero_crossing_rate,
            spectral_centroid,
            spectral_rolloff
        ])
    
    def is_speech_ml(self, audio_frame):
        """ML-based speech detection"""
        
        features = self.extract_vad_features(audio_frame)
        prediction = self.model.predict(features.reshape(1, -1))
        
        return prediction[0] > 0.5
```

### OpenAI Realtime API Integration

#### WebRTC with OpenAI Realtime

```python
import openai
import asyncio
import json
from aiortc import RTCPeerConnection, RTCSessionDescription

class OpenAIRealtimeVAD:
    def __init__(self, api_key):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.session = None
        self.websocket = None
    
    async def create_realtime_session(self):
        """Create OpenAI Realtime API session"""
        
        # Create session with OpenAI Realtime API
        response = await self.client.realtime.session.create(
            model="gpt-4o-realtime-preview",
            modalities=["audio", "text"],
            instructions="You are a voicemail detection assistant. Analyze incoming audio and determine if it's a voicemail or live person.",
        )
        
        self.session = response.session
        
        # Connect to WebSocket
        self.websocket = await websockets.connect(
            f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview",
            extra_headers={
                "Authorization": f"Bearer {self.client.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
        )
        
        return self.session
    
    async def process_audio_for_voicemail_detection(self, audio_stream):
        """Process audio stream for voicemail detection using OpenAI"""
        
        if not self.websocket:
            await self.create_realtime_session()
        
        # Send audio configuration
        await self.websocket.send(json.dumps({
            "type": "input_audio_buffer.clear"
        }))
        
        # Stream audio data
        async for audio_chunk in audio_stream:
            # Convert to base64
            audio_base64 = base64.b64encode(audio_chunk.tobytes()).decode()
            
            await self.websocket.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }))
            
            # Commit audio buffer periodically
            await self.websocket.send(json.dumps({
                "type": "input_audio_buffer.commit"
            }))
            
            # Request response
            await self.websocket.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": "Analyze the audio and respond with 'VOICEMAIL' if this sounds like a voicemail system or 'HUMAN' if it sounds like a live person. Be very brief."
                }
            }))
            
            # Wait for response
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("type") == "response.text.done":
                text_response = response_data["text"]
                
                is_voicemail = "VOICEMAIL" in text_response.upper()
                confidence = 0.9 if is_voicemail else 0.1
                
                return {
                    'is_voicemail': is_voicemail,
                    'confidence': confidence,
                    'method': 'openai_realtime',
                    'raw_response': text_response
                }
        
        return {'is_voicemail': False, 'confidence': 0.0}
```

### Performance Optimization

#### Low-Latency Audio Buffer Management

```python
import collections
from threading import Lock

class OptimizedAudioBuffer:
    def __init__(self, max_size=32000, sample_rate=8000):
        self.max_size = max_size
        self.sample_rate = sample_rate
        self.buffer = collections.deque(maxlen=max_size)
        self.lock = Lock()
        
        # Pre-allocated arrays for efficiency
        self.analysis_array = np.zeros(16000, dtype=np.float32)  # 2 seconds
        self.temp_array = np.zeros(4096, dtype=np.float32)
    
    def add_audio(self, audio_chunk):
        """Add audio chunk to buffer (thread-safe)"""
        
        with self.lock:
            # Convert to list for deque efficiency
            if isinstance(audio_chunk, np.ndarray):
                audio_list = audio_chunk.tolist()
            else:
                audio_list = list(audio_chunk)
            
            self.buffer.extend(audio_list)
    
    def get_analysis_window(self, window_size=16000):
        """Get audio window for analysis (optimized)"""
        
        with self.lock:
            if len(self.buffer) < window_size:
                return None
            
            # Copy to pre-allocated array
            for i, sample in enumerate(list(self.buffer)[-window_size:]):
                self.analysis_array[i] = sample
            
            return self.analysis_array[:window_size].copy()
    
    def clear_old_audio(self, keep_samples=4000):
        """Clear old audio, keeping recent samples"""
        
        with self.lock:
            if len(self.buffer) > keep_samples:
                # Keep only recent samples
                recent_samples = list(self.buffer)[-keep_samples:]
                self.buffer.clear()
                self.buffer.extend(recent_samples)

# High-performance audio processing pipeline
class AudioProcessingPipeline:
    def __init__(self, detector_model, max_latency_ms=100):
        self.detector = detector_model
        self.max_latency = max_latency_ms / 1000.0
        self.sample_rate = 8000
        
        # Processing stages
        self.stages = [
            self.stage_preprocessing,
            self.stage_feature_extraction,
            self.stage_inference,
            self.stage_postprocessing
        ]
        
        # Pre-allocated memory
        self.feature_buffer = np.zeros(512, dtype=np.float32)
        self.mel_spec_buffer = np.zeros((128, 32), dtype=np.float32)
    
    async def process_audio_optimized(self, audio_data):
        """Optimized audio processing pipeline"""
        
        start_time = time.time()
        
        # Process through stages
        data = audio_data
        for stage in self.stages:
            data = await stage(data)
            
            # Check latency constraint
            if time.time() - start_time > self.max_latency:
                return {
                    'is_voicemail': False,
                    'confidence': 0.0,
                    'error': 'latency_exceeded'
                }
        
        processing_time = time.time() - start_time
        
        return {
            'is_voicemail': data['prediction'] > 0.5,
            'confidence': float(data['prediction']),
            'processing_time': processing_time,
            'method': 'optimized_pipeline'
        }
    
    async def stage_preprocessing(self, audio_data):
        """Preprocessing stage"""
        
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Apply windowing
        window = np.hanning(len(audio_data))
        windowed_audio = audio_data * window
        
        return windowed_audio
    
    async def stage_feature_extraction(self, audio_data):
        """Feature extraction stage"""
        
        # Compute mel spectrogram using pre-allocated buffer
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_mels=128
        )
        
        # Reuse buffer if possible
        if mel_spec.shape == self.mel_spec_buffer.shape:
            np.copyto(self.mel_spec_buffer, mel_spec)
            return self.mel_spec_buffer
        else:
            return mel_spec
    
    async def stage_inference(self, features):
        """Model inference stage"""
        
        # Reshape for model input
        model_input = features.reshape(1, *features.shape)
        
        # Run inference
        prediction = self.detector.predict(model_input)
        
        return {'prediction': prediction[0][0], 'features': features}
    
    async def stage_postprocessing(self, data):
        """Postprocessing stage"""
        
        # Apply confidence calibration
        raw_prediction = data['prediction']
        calibrated_confidence = self.calibrate_confidence(raw_prediction)
        
        return {
            'prediction': calibrated_confidence,
            'raw_prediction': raw_prediction
        }
    
    def calibrate_confidence(self, raw_score):
        """Calibrate model confidence score"""
        
        # Simple sigmoid calibration
        # In practice, this would be trained on validation data
        return 1.0 / (1.0 + np.exp(-5 * (raw_score - 0.5)))
```

### Memory-Efficient Processing

#### Streaming Audio with Memory Constraints

```python
class MemoryEfficientProcessor:
    def __init__(self, max_memory_mb=100):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_usage = 0
        self.audio_chunks = {}
        self.processing_queue = asyncio.Queue(maxsize=10)
    
    async def process_audio_stream_memory_efficient(self, audio_stream, call_id):
        """Process audio stream with memory constraints"""
        
        chunk_id = 0
        detection_result = None
        
        async for audio_chunk in audio_stream:
            # Check memory usage
            chunk_size = audio_chunk.nbytes
            
            if self.current_memory_usage + chunk_size > self.max_memory_bytes:
                # Free oldest chunks
                await self.free_memory()
            
            # Store chunk
            self.audio_chunks[f"{call_id}_{chunk_id}"] = audio_chunk
            self.current_memory_usage += chunk_size
            
            # Process when we have enough data
            if chunk_id >= 10:  # ~2 seconds of audio
                detection_result = await self.process_chunks(call_id, chunk_id - 9, chunk_id)
                
                if detection_result['is_voicemail']:
                    break
            
            chunk_id += 1
        
        # Cleanup memory for this call
        await self.cleanup_call_data(call_id)
        
        return detection_result or {'is_voicemail': False, 'confidence': 0.0}
    
    async def free_memory(self):
        """Free memory by removing old chunks"""
        
        # Sort chunks by age and remove oldest
        sorted_chunks = sorted(self.audio_chunks.keys())
        
        while (self.current_memory_usage > self.max_memory_bytes * 0.7 and 
               sorted_chunks):
            
            chunk_key = sorted_chunks.pop(0)
            chunk = self.audio_chunks.pop(chunk_key)
            self.current_memory_usage -= chunk.nbytes
    
    async def process_chunks(self, call_id, start_chunk, end_chunk):
        """Process a range of audio chunks"""
        
        # Combine chunks
        chunks_to_process = []
        for i in range(start_chunk, end_chunk + 1):
            chunk_key = f"{call_id}_{i}"
            if chunk_key in self.audio_chunks:
                chunks_to_process.append(self.audio_chunks[chunk_key])
        
        if not chunks_to_process:
            return {'is_voicemail': False, 'confidence': 0.0}
        
        # Concatenate audio
        combined_audio = np.concatenate(chunks_to_process)
        
        # Run detection
        detector = VoicemailDetector()
        result = await detector.detect_async(combined_audio)
        
        return result
```

## Integration with FreeSWITCH

### Real-Time Audio Streaming from FreeSWITCH

```python
class FreeSWITCHAudioStreamer:
    def __init__(self, freeswitch_host='localhost', esl_port=8021, esl_password='ClueCon'):
        self.fs_host = freeswitch_host
        self.esl_port = esl_port
        self.esl_password = esl_password
        self.active_streams = {}
    
    async def start_audio_streaming(self, call_uuid, detection_callback):
        """Start streaming audio from FreeSWITCH call"""
        
        # Connect to FreeSWITCH ESL
        esl_connection = ESL.ESLconnection(
            self.fs_host, self.esl_port, self.esl_password
        )
        
        if not esl_connection.connected():
            raise Exception("Failed to connect to FreeSWITCH ESL")
        
        # Set up audio streaming
        stream_url = f"ws://localhost:8080/audio/{call_uuid}"
        
        # Configure FreeSWITCH to stream audio
        result = esl_connection.api(
            f"uuid_broadcast {call_uuid} vg_tap_ws::start::{stream_url}"
        )
        
        if "+OK" not in result.getBody():
            raise Exception(f"Failed to start audio streaming: {result.getBody()}")
        
        # Start WebSocket server to receive audio
        audio_server = AudioStreamReceiver(detection_callback)
        await audio_server.start_server(call_uuid)
        
        return stream_url

class AudioStreamReceiver:
    def __init__(self, detection_callback):
        self.detection_callback = detection_callback
        self.detector = VoicemailAIDetector()
    
    async def start_server(self, call_uuid):
        """Start WebSocket server to receive audio from FreeSWITCH"""
        
        async def handle_audio_stream(websocket, path):
            buffer = AudioBuffer()
            
            async for message in websocket:
                # Convert audio bytes to numpy array
                audio_data = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32767.0
                
                buffer.add_audio(audio_data)
                
                # Process when buffer has enough data
                analysis_window = buffer.get_analysis_window()
                if analysis_window is not None:
                    result = await self.detector.detect_async(analysis_window)
                    
                    if result['is_voicemail']:
                        await self.detection_callback(call_uuid, result)
                        break
        
        # Start WebSocket server
        server = await websockets.serve(
            handle_audio_stream,
            "localhost",
            8080
        )
        
        await server.wait_closed()
```

## Performance Monitoring

### Real-Time Performance Metrics

```python
class AudioProcessingMetrics:
    def __init__(self):
        self.metrics = {
            'processing_times': [],
            'memory_usage': [],
            'detection_accuracy': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
            'throughput': {'calls_per_second': 0, 'audio_mb_per_second': 0}
        }
        self.start_time = time.time()
    
    def record_processing_time(self, processing_time):
        """Record processing time for latency monitoring"""
        self.metrics['processing_times'].append(processing_time)
        
        # Keep only recent measurements
        if len(self.metrics['processing_times']) > 1000:
            self.metrics['processing_times'] = self.metrics['processing_times'][-1000:]
    
    def get_real_time_stats(self):
        """Get current performance statistics"""
        
        if not self.metrics['processing_times']:
            return {}
        
        times = self.metrics['processing_times']
        
        return {
            'avg_processing_time': np.mean(times),
            'p95_processing_time': np.percentile(times, 95),
            'p99_processing_time': np.percentile(times, 99),
            'max_processing_time': max(times),
            'current_memory_mb': self.get_memory_usage_mb(),
            'calls_processed': len(times),
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
    
    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
```

## Conclusion

Real-time audio processing for voicemail detection requires careful consideration of:

1. **Latency Requirements**: Target <100ms for telephony applications
2. **Memory Management**: Efficient buffer management for streaming audio
3. **Processing Optimization**: GPU acceleration and batch processing
4. **Integration Complexity**: WebRTC, ESL, and custom protocols
5. **Performance Monitoring**: Real-time metrics and alerting

Key technologies and frameworks:
- **aiortc**: Python WebRTC implementation
- **TensorFlow/PyTorch**: GPU-accelerated ML inference
- **WebRTC VAD**: Voice activity detection
- **OpenAI Realtime API**: Cloud-based AI processing
- **FreeSWITCH ESL**: Real-time call control

Success factors for production deployment:
- Choose appropriate technology stack based on latency requirements
- Implement proper error handling and fallback mechanisms
- Monitor performance metrics continuously
- Optimize memory usage for high-volume scenarios
- Use connection pooling and batch processing where possible

The combination of these technologies enables the development of highly responsive voicemail detection systems capable of processing real-time audio streams with minimal latency while maintaining high accuracy.