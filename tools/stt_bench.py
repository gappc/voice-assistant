"""Benchmark speech-to-text models on your own recordings.

    uv run tools/stt_bench.py record [--device N]   # read prompts into the mic
    uv run --with faster-whisper --with 'onnx-asr[hub]' --with sherpa-onnx tools/stt_bench.py run [--models ...] [--threads N]

`record` saves raw (unconditioned) 16 kHz mono clips plus their reference
text. `run` loads each model in its own subprocess, so peak RAM is measured
per model, feeds every clip through the app's _condition_audio(), and reports
load time, latency, CPU time, peak RAM and word error rate against the prompt.

Parakeet v3 runs on three runtimes: onnx-asr, sherpa-onnx and parakeet.cpp
(the app's speech_recognizer loader, with the CPU or Vulkan build forced). Models
and libraries the pip packages don't fetch themselves are downloaded to
~/.cache/voice-assistant/ on first use.
"""
import argparse
import ctypes
import json
import os
import re
import resource
import subprocess
import sys
import tarfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLERATE = 16000
DEFAULT_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "voice-assistant" / "stt-bench"

PROMPTS = [
    ("en", "Thanks, that works for me."),
    ("en", "Can you send me the report by Friday afternoon?"),
    ("en", "Open a pull request against main and tag the reviewer."),
    ("en", "The meeting moved to half past three, so I'll be about ten minutes late."),
    ("en", "Restart the Kubernetes pod and check whether the latency drops below two hundred milliseconds."),
    ("de", "Danke, das passt mir gut."),
    ("de", "Kannst du mir die Unterlagen bis Freitag schicken?"),
    ("de", "Ich bin heute im Homeoffice und ab vierzehn Uhr erreichbar."),
    ("de", "Die Rechnung für den Oktober ist leider noch nicht bezahlt worden."),
    ("de", "Bitte prüfe, ob der Server nach dem Update wieder normal läuft."),
]

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "voice-assistant"
SHERPA_MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2")
PCPP_MODEL_URL = "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/{file}"

# name -> (backend, model id, extra options)
MODELS = {
    "whisper-base":     ("faster-whisper", "base", {}),
    "whisper-small":    ("faster-whisper", "small", {}),
    "whisper-turbo":    ("faster-whisper", "large-v3-turbo", {}),
    "parakeet-v3":      ("onnx-asr", "nemo-parakeet-tdt-0.6b-v3", {"quantization": "int8"}),
    "canary-1b-v2":     ("onnx-asr", "nemo-canary-1b-v2", {"quantization": "int8"}),
    "sherpa-v3":        ("sherpa-onnx", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", {}),
    "pcpp-v3-q8":       ("parakeet.cpp", "tdt-0.6b-v3-q8_0.gguf", {"variant": "cpu"}),
    "pcpp-v3-q4":       ("parakeet.cpp", "tdt-0.6b-v3-q4_k.gguf", {"variant": "cpu"}),
    "pcpp-v3-vulkan":   ("parakeet.cpp", "tdt-0.6b-v3-q8_0.gguf", {"variant": "vulkan"}),
}
DEFAULT_MODELS = ["whisper-base", "parakeet-v3", "sherpa-v3", "pcpp-v3-q8", "pcpp-v3-q4", "pcpp-v3-vulkan"]


# --- Recording -----------------------------------------------------------

def write_wav(path, audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLERATE)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32767


def record(args):
    import sounddevice as sd

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    print(f"Input device: {sd.query_devices(device, 'input')['name']}")
    print(f"Saving to {out}\nFor each prompt: Enter to start, Enter to stop, "
          "'s' + Enter to skip, 'q' + Enter to quit.\n")

    refs = {}
    refs_path = out / "references.json"
    if refs_path.exists():
        refs = json.loads(refs_path.read_text())

    for i, (lang, text) in enumerate(PROMPTS):
        name = f"{i:02d}_{lang}.wav"
        print(f"[{i + 1}/{len(PROMPTS)}] ({lang}) {text}")
        cmd = input("  ready? ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            continue
        frames = []
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype="float32",
                            device=device, callback=lambda d, *_: frames.append(d.copy())):
            input("  recording... Enter to stop ")
        audio = np.concatenate(frames)[:, 0] if frames else np.zeros(0, np.float32)
        write_wav(out / name, audio)
        refs[name] = {"lang": lang, "text": text}
        refs_path.write_text(json.dumps(refs, indent=2, ensure_ascii=False))
        print(f"  saved {name} ({len(audio) / SAMPLERATE:.1f}s)\n")


# --- Scoring -------------------------------------------------------------

def normalize(text):
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def wer(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1), len(r)


# --- Downloads -----------------------------------------------------------

def fetch(url, dest):
    """Download `url` to `dest` unless it is already there."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...", file=sys.stderr, flush=True)
    tmp = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def fetch_tarball(url, dest_dir):
    """Download and unpack `url` into `dest_dir` unless already unpacked."""
    if dest_dir.exists():
        return dest_dir
    archive = fetch(url, dest_dir.with_name(url.rsplit("/", 1)[1]))
    with tarfile.open(archive) as tar:
        tar.extractall(dest_dir.with_name(dest_dir.name + ".part"), filter="data")
    dest_dir.with_name(dest_dir.name + ".part").rename(dest_dir)
    archive.unlink()
    return dest_dir


# --- Benchmark worker (one model per process) ----------------------------

def rss_mb():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


def worker(args):
    from voice_assistant import _condition_audio

    backend, model_id, opts = MODELS[args.worker]
    clips = json.loads(Path(args.dir, "references.json").read_text())
    base_rss = rss_mb()

    t0 = time.perf_counter()
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel(model_id, device="cpu", compute_type="int8", cpu_threads=args.threads)

        def transcribe(audio, lang):
            segments, _ = model.transcribe(audio, beam_size=5, language=lang, vad_filter=True)
            return " ".join(s.text for s in segments).strip()
    elif backend == "sherpa-onnx":
        import sherpa_onnx
        d = fetch_tarball(SHERPA_MODEL_URL, CACHE_DIR / "sherpa-onnx" / model_id) / model_id
        # sherpa-onnx defaults to a single thread; give it one per physical core
        # so "library default" compares like with like.
        model = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
            num_threads=args.threads or max(os.cpu_count() // 2, 1), model_type="nemo_transducer")

        def transcribe(audio, lang):
            stream = model.create_stream()
            stream.accept_waveform(SAMPLERATE, audio)
            model.decode_stream(stream)
            return stream.result.text.strip()
    elif backend == "parakeet.cpp":
        import speech_recognizer
        # The app's loader: same pinned, checksum-verified release, one variant forced.
        lib = speech_recognizer.load_library(CACHE_DIR / "parakeet-cpp", variants=(opts["variant"],))
        gguf = fetch(PCPP_MODEL_URL.format(file=model_id), CACHE_DIR / "parakeet-cpp" / "models" / model_id)
        ctx = lib.parakeet_capi_load(str(gguf).encode())
        if not ctx:
            sys.exit(f"parakeet.cpp load failed: {lib.parakeet_capi_load_error().decode()}")
        if args.threads:
            # Not in the C API; the C++ setter is exported and applies to later calls.
            lib._ZN2pk15set_num_threadsEi(args.threads)

        def transcribe(audio, lang):
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            out = lib.parakeet_capi_transcribe_pcm(ctx, audio.ctypes.data, len(audio), SAMPLERATE, 0)
            text = ctypes.string_at(out).decode()
            lib.parakeet_capi_free_string(out)
            return text.strip()
    else:
        import onnx_asr
        import onnxruntime as ort
        sess = ort.SessionOptions()
        if args.threads:
            sess.intra_op_num_threads = args.threads
        model = onnx_asr.load_model(model_id, sess_options=sess, **opts)
        takes_lang = "canary" in model_id

        def transcribe(audio, lang):
            kw = {"language": lang} if takes_lang else {}
            return model.recognize(audio, sample_rate=SAMPLERATE, **kw).strip()
    load_s = time.perf_counter() - t0

    # Warm-up so first-call graph setup isn't billed to the first clip.
    transcribe(np.zeros(SAMPLERATE, np.float32), "en")

    results = []
    for name, ref in sorted(clips.items()):
        audio, _, _ = _condition_audio(read_wav(Path(args.dir, name)))
        w0, c0 = time.perf_counter(), time.process_time()
        hyp = transcribe(audio, ref["lang"])
        results.append({
            "clip": name, "lang": ref["lang"], "ref": ref["text"], "hyp": hyp,
            "seconds": len(audio) / SAMPLERATE,
            "wall": time.perf_counter() - w0, "cpu": time.process_time() - c0,
        })

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"load_s": load_s, "model_mb": peak - base_rss, "peak_mb": peak,
                      "results": results}))


# --- Driver --------------------------------------------------------------

def run(args):
    if not Path(args.dir, "references.json").exists():
        sys.exit(f"No clips in {args.dir}; run `record` first.")

    report = {}
    for name in args.models:
        print(f"== {name} (first run downloads the model) ...", file=sys.stderr, flush=True)
        cmd = [sys.executable, __file__, "--dir", args.dir, "--worker", name]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"   failed:\n{proc.stderr[-2000:]}", file=sys.stderr)
            continue
        report[name] = json.loads(proc.stdout.strip().splitlines()[-1])

    print("\n## Transcripts")
    clip_names = sorted({r["clip"] for m in report.values() for r in m["results"]})
    for clip in clip_names:
        first = next(r for m in report.values() for r in m["results"] if r["clip"] == clip)
        print(f"\n{clip}  REF: {first['ref']}")
        for name, m in report.items():
            r = next(r for r in m["results"] if r["clip"] == clip)
            print(f"  {name:14s} {wer(r['ref'], r['hyp'])[0]:5.0%}  {r['hyp']}")

    print("\n## Summary (WER = word errors / reference words, pooled per language)")
    print(f"{'model':14s} {'WER en':>7s} {'WER de':>7s} {'avg lat':>8s} {'max lat':>8s} "
          f"{'CPU/audio':>9s} {'load':>6s} {'model RAM':>10s} {'peak RAM':>9s}")
    for name, m in report.items():
        res = m["results"]
        werr = {}
        for lang in ("en", "de"):
            errs = [(wer(r["ref"], r["hyp"])) for r in res if r["lang"] == lang]
            words = sum(n for _, n in errs)
            werr[lang] = sum(e * n for e, n in errs) / words if words else float("nan")
        lat = [r["wall"] for r in res]
        cpu_ratio = sum(r["cpu"] for r in res) / sum(r["seconds"] for r in res)
        print(f"{name:14s} {werr['en']:7.1%} {werr['de']:7.1%} {np.mean(lat):7.2f}s {max(lat):7.2f}s "
              f"{cpu_ratio:8.2f}x {m['load_s']:5.1f}s {m['model_mb']:8.0f}MB {m['peak_mb']:7.0f}MB")
    print("\nCPU/audio: CPU-seconds spent per second of speech (summed over all threads).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", choices=["record", "run"])
    p.add_argument("--dir", default=str(DEFAULT_DIR), help="clip directory")
    p.add_argument("--device", help="input device index or name (record)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS))
    p.add_argument("--threads", type=int, default=0, help="cap CPU threads (0 = library default, one per physical core for sherpa-onnx)")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        worker(args)
    elif args.command == "record":
        record(args)
    elif args.command == "run":
        run(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
