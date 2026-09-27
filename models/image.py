import shutil
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor
from typing import List

SAFETY_MODEL = "CompVis/stable-diffusion-safety-checker"
NEGATIVE_PROMPT = (
    "nsfw, nude, naked, nudity, sexual, explicit, lingerie, cleavage, "
    "gore, blood, deformed, watermark, text"
)
MAX_ATTEMPTS = 3
FILLER_IMAGE = "images/filler.png"


class ImageModel:
    def __init__(self, storage_path: str = ""):
        self.storage_path = storage_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"🔄 Loading image model on {device}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            torch_dtype=dtype,
            # the model repo ships without a checker, so bring the standard one
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                SAFETY_MODEL, dtype=dtype
            ),
            feature_extractor=CLIPImageProcessor.from_pretrained(SAFETY_MODEL),
        ).to(device)
        print("✅ Image Model loaded")

        self.generator = torch.Generator(device).manual_seed(42)

    def generate_image(self, prompt: str, filename: str):
        outfile_path = f"{self.storage_path}{filename}.png"

        # generator advances each call, so a retry is a fresh seed
        for attempt in range(1, MAX_ATTEMPTS + 1):
            out = self.pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=848,
                width=480,
                num_inference_steps=40,
                generator=self.generator,
            )
            if not out.nsfw_content_detected[0]:
                out.images[0].save(outfile_path)
                print(f"✅ Image saved as {outfile_path}")
                return
            print(f"⚠️ NSFW flagged (attempt {attempt}/{MAX_ATTEMPTS}): {prompt}")

        shutil.copy(FILLER_IMAGE, outfile_path)
        print(f"⚠️ Using filler image for {outfile_path}")

    def generate_images(self, prompts: List[str], filename_prefix: str):
        for i, prompt in enumerate(prompts, start=1):
            self.generate_image(prompt, f"{filename_prefix}_{i:02d}")
