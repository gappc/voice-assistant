# Dictation moves from Whisper base to Parakeet v3 on the GPU

**Status:** done 2026-10-14 (`speech_recognizer.py`)
**Replaces:** faster-whisper `base`, int8, CPU

## Why

Whisper `base` misheard technical words and short German phrases ("qonatis
port" for "Kubernetes pod", "mit dem Brüfer" for "Bitte prüfe"), and its
language auto-detection was unreliable enough that dictation needed a tray
language picker. Bigger Whisper models fix accuracy but cost seconds per phrase
on this CPU (large-v3-turbo: ~5 s).

## What we measured

`tools/stt_bench.py` on 20 real phrases (10 English, 10 German), all recorded
on the laptop's quiet built-in mic (DC offset ~0.16–0.25, boosted 1–6x by
`_condition_audio()`). Ryzen AI 7 350, Radeon 860M. "Real errors" excludes
formatting such as "10" for "ten"; second recording set shown.

| model / runtime              | real errors en / de | latency | CPU per s of audio | memory |
|------------------------------|---------------------|---------|--------------------|--------|
| Whisper base, faster-whisper | 6 / 4               | 0.38 s  | 0.81x              | 420 MB |
| Whisper small, faster-whisper| 0 / 1               | 0.91 s  | 1.38x              | 700 MB |
| Parakeet v3, onnx-asr int8   | dropped one phrase / 0 | 0.29 s | 1.21x            | 1.3 GB |
| Parakeet v3, parakeet.cpp q4, CPU | 2 / 0          | 0.16 s  | 0.34x              | 800 MB |
| **Parakeet v3, parakeet.cpp q4, Vulkan** | **1 / 0** | **0.09 s** | **0.01x** | ~250 MB + ~740 MB GPU |

Findings worth keeping:

- **The runtime matters, not just the model.** parakeet.cpp is validated
  against NeMo's own output and beat the int8 ONNX exports of the same model
  (onnx-asr, sherpa-onnx) on both recording sets.
- **The Radeon iGPU is usable through ggml's Vulkan backend** (RADV), even
  though onnxruntime has no path to it. On an iGPU, "GPU memory" is system RAM,
  so Vulkan and CPU need about the same memory in total.
- **q4_k transcribed identically to q8_0** with ~250 MB less memory.
- **No VAD needed:** Parakeet returns "" for digital silence, DC-offset noise,
  room noise and key clicks.
- **The NPU (XDNA 2) is not worth it yet:** AMD's Ryzen AI STT stack is
  Windows-only; FastFlowLM runs Whisper turbo on it at ~0.2x real time,
  slower than Vulkan Parakeet.

## How it works

- `speech_recognizer.py` loads parakeet.cpp's prebuilt C library (v0.6.1) with
  ctypes. The Vulkan build uses the first GPU and falls back to the CPU by
  itself; the CPU build is only used when `libvulkan.so.1` is missing.
  `PARAKEET_DEVICE=cpu` forces the CPU.
- Library and model are downloaded on first start to
  `~/.local/share/voice-assistant/stt/`, pinned and SHA-256-checked.
- Parakeet detects the language per recording, so the Dictation Language
  tray menu and its `stt_language` setting were removed.

## Open

- The BRIO has not been benchmarked; both recording sets came from the
  built-in mic. A cleaner mic should be the easier case.
- Thread count on the CPU fallback is the library default (all physical
  cores, spinning briefly); the C API has no setter.
