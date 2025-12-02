# SER + FER — Speech Emotion Recognition & Face Emotion Recognition (Real-Time, Advanced Edition)

> **Professional guide** for building a production-grade, low-latency multimodal emotion system with live 3D avatar visualization. This version replaces playful tone with a professional, implementation-first style and adds: 3D avatar frontends controllable by backend emotion outputs, integration with **SenseVoice** for SER, concrete data contracts, and engineering-level notes for deployment and optimization.

---

## Table of contents

1. [Executive summary](#executive-summary)
2. [High-level goals & success metrics](#goals)
3. [System overview & real-time architecture](#overview)
4. [Protocols & data contracts (messages)](#protocols)
5. [Frontend 3D avatar integration](#frontend-3d)
6. [SER: SenseVoice integration & audio pipeline](#ser-sensevoice)
7. [FER: face tracking, AU extraction & head pose](#fer-advanced)
8. [Multimodal fusion → avatar driving](#fusion-to-avatar)
9. [Model serving, ops, and latency optimizations](#ops)
10. [Privacy, security & consent engineering](#privacy)
11. [Datasets, annotation, and live fine-tuning strategy](#datasets)
12. [Monitoring, evaluation, and A/B experimentation](#monitoring)
13. [Roadmap & phased delivery plan](#roadmap)
14. [Appendix: schema examples, mapping tables, implementation notes]

---

## 1. Executive summary <a name="executive-summary"></a>

This document describes a production-ready pipeline to infer user emotion continuously from live audio and camera streams (SER + FER) and drive **interactive 3D avatars** in the frontend. The intended product provides low-latency, privacy-preserving emotion feedback and expressive avatar control for applications such as virtual assistants, telepresence, educational software, game NPCs, and research tools.

Key differentiators:

* SER powered by **SenseVoice** for robust, low-latency speech emotion cues.
* FER pipeline provides emotion probabilities, action-unit estimates, head pose, and eye-gaze.
* Standardized JSON+binary streaming contracts for deterministic avatar control.
* Avatar control outputs (blendshapes, bone transforms, viseme triggers) to produce natural, low-jitter animation.
* End-to-end latency targets: **Phase 0: <200ms**, **Phase 1: <100ms** for on-prem GPU clusters or edge-accelerated instances.

---

## 2. Goals & success metrics <a name="goals"></a>

**Functional goals**

* Continuous emotion inference from live microphone + front camera.
* 3D avatar updates at interactive frame rates (30–60 FPS) while receiving emotion updates at 10–60 Hz.
* Lip-sync/viseme control derived from audio (SenseVoice or equivalent).
* Robustness across devices, lighting, and noisy environments.

**Success metrics**

* FER top-1 accuracy on held-out test: ≥75% (AffectNet-like domain adaptation).
* SER classification F1: ≥0.70 on in-domain tests (SenseVoice fine-tune + augmentation).
* Avatar perceptual quality: mean rating ≥4/5 in user studies for emotional fidelity.
* End-to-end 95th percentile latency: <200ms (Phase 0) and <100ms (Phase 1).
* Jitter (avatar parameter variance between consecutive frames after smoothing): <5%.

---

## 3. System overview & real-time architecture <a name="overview"></a>

### Components

* **Client (Mobile/Web/VR)** — captures camera + mic, hosts 3D renderer (Three.js / React Three Fiber / Babylon / Unity / Unreal), and receives avatar instructions.
* **Gateway / Ingest** — WebRTC or WebSocket gateway that receives streams and forwards to inference services.
* **Inference Cluster** — model servers for FER (vision) and SER (SenseVoice + optional fine-tuned models). Prefer Triton/ONNXRuntime with GPU acceleration.
* **Fusion Service** — attention-weighted fusion of SER & FER embeddings → canonical emotion vector + avatar driving parameters.
* **State & Pub/Sub** — Redis for ephemeral session state, Kafka/NATS for scaling updates.
* **Analytics & Monitoring** — Prometheus + Grafana, plus custom perceptual telemetry.

### Data flow

1. Client opens a low-latency media session (WebRTC) with data channel or WebSocket fallback.
2. Client sends periodic frames (or full media) to Gateway; audio frames are sent as PCM or Opus.
3. Gateway forwards audio frames to SenseVoice + local SER models; forwards images to FER model worker.
4. Inference workers push time-stamped emotion features to Fusion Service.
5. Fusion Service computes avatar control parameters and publishes them to client via the same data channel.
6. Client applies smoothing and drives the 3D avatar in real time.

**Latency optimization points:** WebRTC for transport, model quantization (FP16/INT8), batching windows sized to balance latency vs throughput, and data channel compression for avatar messages.

---

## 4. Protocols & data contracts (messages) <a name="protocols"></a>

Use compact, typed JSON or binary protobuf messages on a persistent data channel. Include `session_id`, `timestamp_ms`, and `sequence` to allow client reordering & interpolation.

**Example JSON (emotion update):**

```json
{
  "type": "emotion_update",
  "session_id": "abc-123",
  "timestamp_ms": 1700000123456,
  "sequence": 234,
  "payload": {
    "emotion": "happy",
    "scores": {"happy": 0.78, "neutral": 0.12, "sad": 0.05, "angry": 0.03},
    "valence": 0.64,
    "arousal": 0.32,
    "au": {"au12_smile": 0.74, "au06_cheek_raise": 0.42},
    "head_pose": {"yaw": 3.2, "pitch": -1.1, "roll": 0.2},
    "gaze": {"x": 0.02, "y": -0.01},
    "speaking_probability": 0.86
  }
}
```

**Avatar control message (reduced to the minimal fields for frame-rate):**

```json
{
  "type": "avatar_frame",
  "session_id": "abc-123",
  "timestamp_ms": 1700000123457,
  "sequence": 235,
  "payload": {
    "blendshapes": {"smile": 0.78, "brow_up": 0.12, "frown": 0.02},
    "bone_transforms": {"neck_pitch": -1.1, "head_yaw": 3.2},
    "viseme": "VV5",
    "viseme_conf": 0.88,
    "particles": {"glow_intensity": 0.2}
  }
}
```

**Notes:** message size should be kept small; use integer quantization or CBOR/protobuf for production.

---

## 5. Frontend 3D avatar integration <a name="frontend-3d"></a>

### Rendering platforms & libraries

* **Web**: Three.js with React Three Fiber (r3f) or Babylon.js.
* **Mobile**: Unity (via flutter_unity_widget or native), Unreal, or native OpenGL/Metal via SceneKit (iOS) or Filament (Android).
* **Cross-platform**: Unity provides fastest iteration for expressive avatars; r3f is excellent for web UIs and prototypes.

### Avatar rigging & asset format

* Use GLTF 2.0 for web-friendly assets.
* Rig must expose **blendshape/morph target** controls for facial expressions and **bones** for head/neck/upper-body motion.
* Provide viseme blendshapes or phoneme-to-viseme mapping.

### Real-time control loop (client)

1. Receive `avatar_frame` messages on data channel.
2. Apply smoothing (EMA or critically-damped spring) to each parameter.
3. Map normalized values to blendshape/morph target weights (0–1).
4. Apply bone rotations with SLERP for continuity.
5. Run the renderer at 60 FPS and update visuals; avatar parameters can update at 10–60 Hz.

### Lip-sync & visemes

* Use SenseVoice phoneme/viseme outputs where available, or compute audio energy → viseme fallback.
* Trigger viseme blendshape transitions with short crossfades (30–60 ms) to avoid popping.

### Example client-side libraries & modules

* React: `@react-three/fiber`, `three/examples/jsm/loaders/GLTFLoader`, `drei` utilities.
* Unity: use `PlayableGraph` and `Animation Rigging` to map blendshapes & bone transforms.
* Flutter: `flutter_unity_widget` for Unity integration or `flutter_gl` + custom shader pipeline for GLTF rendering.

---

## 6. SER: SenseVoice integration & audio pipeline <a name="ser-sensevoice"></a>

### Why SenseVoice

SenseVoice provides low-latency, production-ready speech emotion features (prosody, sentiment, stress markers). Use it as a primary SER engine for robustness and speed, and complement it with an in-house fine-tuned model for domain adaptation.

### Audio capture & preprocessing

* Capture audio at 16 kHz mono PCM (or 16/24 kHz if higher fidelity is needed).
* Use short overlapping windows (e.g., 1s windows with 50% overlap) for near-instant emotion responsiveness.
* Apply voice activity detection (VAD) to avoid sending silence-heavy payloads.

### Integration pattern

1. Client sends short encoded PCM frames to Gateway (WebRTC datachannel or RPC).
2. Gateway forwards or streams to SenseVoice (via SDK or REST/gRPC endpoint) for real-time emotion signals and phoneme/viseme timestamps.
3. SenseVoice returns: emotional scores, speech activity, phoneme timestamps, speaking probability, and optionally spectral features.
4. Use SenseVoice output as a primary SER signal; fuse with an in-house model if customization is required.

### Latency & throughput

* Configure SenseVoice to return streaming partial hypotheses (low-latency incremental output) if available.
* Use audio chunking to avoid large buffering; recommended max buffer: 500–1000 ms to keep latency low.

### Fallbacks

* If SenseVoice unavailable, run an embedded lightweight SER model on-device (edge) to provide a degraded but immediate experience.

---

## 7. FER: face tracking, AU extraction & head pose <a name="fer-advanced"></a>

### Core outputs

* Per-frame emotion probabilities (neutral, sad, happy, angry, surprise, disgust, fear).
* Facial Action Units (AUs) with intensity values (e.g., AU12, AU06, AU04), useful for direct mapping to blendshapes.
* Head pose: yaw/pitch/roll.
* Eye gaze vector (screen-relative) and blink detection.
* Face tracking ID for multi-face sessions.

### Detection & inference

* Face detection: RetinaFace or BlazeFace for fast detection; use a light tracker to avoid re-detecting every frame.
* FER model: ResNet-50 / EfficientNet variant fine-tuned on AffectNet, RAF-DB.
* AU estimator: small dedicated head to predict AU intensities (can be multi-task joint model with FER).

### On-device vs server

* On-device FER (TFLite or CoreML) reduces network and privacy exposure but may be less accurate.
* Server-side provides best accuracy and simplified model updates. Consider hybrid: lightweight on-device detection + server-side refinement.

---

## 8. Multimodal fusion → avatar driving <a name="fusion-to-avatar"></a>

### Fusion strategy

* **Late fusion with attention**: maintain per-modality embeddings; compute attention weights using confidence + context.
* **Emotion canonicalization**: canonical emotion vector includes `valence`, `arousal`, `dominance`, `speaking_prob`, `AU_map`, `pose`.

### Mapping to avatar parameters

* **AU → blendshapes**: direct mapping (AU12 → smile weight). Use linear mapping with clamping & per-user calibration.
* **Emotion scores → posture & micro-expressions**: e.g., high arousal → slight body lean forward + eye widening.
* **Speaking probability + viseme → mouth shapes**: switch to audio-driven lip-sync during speech.
* **Head pose smoothing**: combine detected head pose with small avatar exaggeration factor.

### Smoothing & temporal filtering

* Use exponential moving averages (α tuned per parameter) or critically-damped spring systems to remove jitter.
* Maintain a short timeline buffer (200–500 ms) to interpolate and compensate for network jitter.

### Consistency checks

* If `speaking_probability` > 0.7, prioritize viseme-based mouth shapes over FER mouth-related AUs.
* Use confidence thresholds to ignore low-confidence modality outputs.

---

## 9. Model serving, ops, and latency optimizations <a name="ops"></a>

### Serving

* Prefer Triton Inference Server or ONNXRuntime for GPU-accelerated model serving.
* Expose gRPC and REST endpoints; use server-side batching for image workloads with micro-batches sized to keep latency low.

### Optimization

* Quantize models to FP16 or INT8 using calibration datasets.
* Use CUDA/cuDNN and TensorRT for the vision stack.
* For SER, use streaming-friendly encoders that support chunk-wise inference (no long context windows).

### Autoscaling & resilience

* Stateless workers behind a service mesh (Istio/linkerd) and autoscale using request latency and queue lengths.
* Use backpressure mechanisms at Gateway to drop frames (gracefully) during overload.

### Edge inference

* For telepresence use-cases, deploy lightweight FER + SER to edge devices (NVIDIA Jetson / Apple Neural Engine / Android NNAPI) to get to <50 ms.

---

## 10. Privacy, security & consent engineering <a name="privacy"></a>

* Require explicit consent flows before enabling camera/audio emotion streams.
* Do not store raw audio/video without explicit opt-in; prefer storing anonymized embeddings if necessary and legally vetted.
* Encrypt all traffic with TLS 1.3 and end-to-end encryption where feasible.
* Provide a visible recording indicator and a one-click stop for users.
* Implement data retention policies and tools to delete session data on request.

---

## 11. Datasets, annotation, and live fine-tuning strategy <a name="datasets"></a>

* Start with AffectNet, RAF-DB, FER2013 for FER; RAVDESS, IEMOCAP, and in-domain voice datasets for SER.
* Collect opt-in in-app examples for domain adaptation; label via a semi-supervised pipeline or human-in-the-loop annotation for quality.
* Use continual learning with replay buffers; monitor model drift and biases across demographics.

---

## 12. Monitoring, evaluation, and A/B experimentation <a name="monitoring"></a>

* Measure latency (p50/p95), inference confidences, session durations, and user opt-out rates.
* Run perceptual A/B tests for avatar mapping strategies (direct AU mapping vs emotion-driven heuristics).
* Track fairness metrics and false-positive rates for sensitive classes.

---

## 13. Roadmap & phased delivery plan <a name="roadmap"></a>

**Phase 0 (Prototype, 2–6 weeks)**

* React web demo with GLTF avatar (r3f), FastAPI backend, DeepFace/Light FER on server.
* SenseVoice prototype integration for audio.
* Basic fusion → avatar mapping and smoothing.

**Phase 1 (MVP, 2–3 months)**

* WebRTC gateway, Triton-based serving, Redis state store.
* Unity mobile client with avatar rig, viseme support from SenseVoice.
* Edge fallback for degraded connectivity.

**Phase 2 (Scale & polish, 3–6 months)**

* Full ResNet-50 FER + Wav2Vec2 hybrid SER fine-tuned with live data.
* Quantized models, autoscaling cluster, monitoring, and compliance audit.

**Phase 3 (Enterprise-grade, 6–12 months)**

* Real-time personalization (per-user calibration), advanced attention-based fusion, and multi-lingual SER enhancements.

---

## 14. Appendix — schema examples, mapping tables, implementation notes <a name="appendix"></a>

### 14.1 Sample avatar mapping table (AU → blendshape)

| Action Unit | Description               |   Blendshape target |                         Mapping function |
| ----------- | ------------------------- | ------------------: | ---------------------------------------: |
| AU12        | Lip corner puller (smile) |       `blend_smile` | linear: weight = clamp(AU12 * 1.1, 0, 1) |
| AU06        | Cheek raise               | `blend_cheek_raise` |                      linear with damping |
| AU04        | Brow lower                |       `blend_frown` |         sigmoid mapping for smooth onset |

### 14.2 Example pseudocode: client control loop

```js
// receives avatar_frame events and applies smoothing
let state = { blendshapes: {}, bones: {} };
function onAvatarFrame(msg) {
  const payload = msg.payload;
  for (const [k, v] of Object.entries(payload.blendshapes)) {
    state.blendshapes[k] = smooth(state.blendshapes[k] || 0, v, 0.2);
  }
  applyToModel(state);
}

function smooth(prev, target, alpha) {
  return prev * (1 - alpha) + target * alpha;
}
```

### 14.3 Example mapping from emotion → avatar stylistic controls

| Emotion | Avatar effect                                      | Parameters                               |
| ------: | -------------------------------------------------- | ---------------------------------------- |
|   Happy | Larger smile, eye crinkle, light particle sparkles | smile +0.8, au06 +0.4, particle_glow 0.2 |
|     Sad | Slight gaze down, depressed shoulders              | head_pitch +3deg, body_slouch 0.3        |

### 14.4 Recommended libraries & infra

**Frontend**: Three.js, @react-three/fiber, GLTFLoader, Unity 2021+, AnimationRigging

**Backend / Models**: Triton Server, ONNXRuntime, TensorRT, SenseVoice SDK, PyTorch Lightning for training

**Transport & infra**: WebRTC, gRPC, Kafka/NATS, Redis, Prometheus/Grafana

---

### Final notes

This document is intended to be both a technical blueprint and an engineering checklist. Implementation requires close iteration with UX designers and audio/animation artists to refine avatar mappings. The system must be built incrementally: start with a deterministic mapping between AUs & blendshapes, then layer in learned fusion models and personalization.
