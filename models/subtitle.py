import torch
import whisperx
from typing import List


class SubtitleModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "float32"
        # silero: pyannote's VAD checkpoint fails torch>=2.6 weights_only loading
        self.model = whisperx.load_model(
            "medium", device=self.device, compute_type=compute_type, vad_method="silero"
        )

    def generate_subtitle(self, audio_filepath: str) -> List:
        result = self.model.transcribe(audio_filepath)
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device
        )
        aligned = whisperx.align(
            result["segments"], model_a, metadata, audio_filepath, device=self.device
        )

        print("✅ Subtitles generated!")

        return list(aligned["word_segments"])
