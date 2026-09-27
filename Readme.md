# 🎬 Offline AI Video Generator

**Offline AI Video Generator** is a modular pipeline that automatically creates short-form videos using local AI models. It combines multiple components—language generation, text-to-speech, image generation, subtitle alignment, and video rendering—into a fully offline workflow.

Ideal as a demonstration of practical AI integration, media generation, and automation systems in a real-world scenario.

---

## 📌 Key Features

- 🔒 **Fully Offline Workflow**  
  After the first run downloads the models, no internet connection is required.

- 🧠 **Content Generation**  
  Uses a local LLM via [Ollama](https://ollama.com/) (`llama3.2`) to generate the video script and image prompts.

- 🎙 **Text-to-Speech (TTS)**  
  Converts the script into audio using [Coqui TTS](https://github.com/idiap/coqui-ai-TTS).

- 🖼 **Image Generation**  
  Creates images with Stable Diffusion (`SG161222/Realistic_Vision_V5.1_noVAE`), about one per 6 seconds of narration, with a slow Ken Burns zoom.
  Explicit images are blocked: family-friendly LLM prompts, an NSFW negative prompt, and the standard SD safety checker. A flagged image is regenerated up to 3 times, then replaced with `images/filler.png`.

- 📝 **Subtitle Generation**  
  Aligns speech with text using [WhisperX](https://github.com/m-bain/whisperX) for word-level captions; the word being spoken is highlighted.

- 🎞 **Video Composition**  
  Assembles images, audio, subtitles, and background music into a final MP4 with MoviePy.

Runs on an NVIDIA GPU when available, CPU otherwise (slow).

---

## ⚙️ Installation

### Prerequisites

- Python 3.11
- [ffmpeg](https://ffmpeg.org/) on PATH
- [Ollama](https://ollama.com/)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) (phonemizer for the TTS voice). Windows: `winget install eSpeak-NG.eSpeak-NG`, then add `C:\Program Files\eSpeak NG` to PATH

### 1. Set up a virtual environment

```bash
py -3.11 -m venv venv
venv\Scripts\activate  # macOS/Linux: source venv/bin/activate
```

### 2. Install dependencies

```bash
# CUDA build of torch (cu128 supports RTX 50xx); use the CPU index if you have no NVIDIA GPU
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 3. Pull the LLM

```bash
ollama pull llama3.2
```

### 4. Usage

```bash
python main.py                                   # random topic
python main.py "a surprising origin of a modern invention"
OLLAMA_MODEL=gemma3:4b python main.py            # any pulled Ollama model
python main.py --review                          # approve script and image prompts before rendering
```

Output lands in `outputs/<timestamp>/` (`content/`, `audio/`, `images/`, `video.mp4`).

### Tests

```bash
python test_video.py
```
