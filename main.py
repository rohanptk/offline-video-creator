from models.text import TextModel
from models.image import ImageModel
from models.audio import AudioModel
from models.subtitle import SubtitleModel
from models.video import VideoModel
import util
import argparse
import os
import sys
from typing import Callable, List, Optional
import time
import random
import soundfile

SECONDS_PER_IMAGE = 6


DEFAULT_TOPICS = [
    "a historical inspiring fact that changed computer science",
    "an ancient story about a clever king or warrior",
    "a strange or funny fact from world history",
    "a scientific discovery that changed human life",
    "a breakthrough in medical science that saved millions",
    "a dramatic moment from World War 1 or World War 2",
    "a surprising origin of a modern invention",
    "a fun random fact that will make people smile",
]


def reviewed(generate: Callable, label: str, review: bool):
    """Run generate(); with review on, regenerate until the user approves."""
    while True:
        result = generate()
        if not review:
            return result
        print(f"\n📝 {label}:\n{result}\n")
        if input("Approve? [y/N]: ").strip().lower() == "y":
            return result
        print("🔁 Regenerating...")


def create_script(storage: str, topic: str, review: bool) -> str:
    content = reviewed(lambda: TextModel().generate_content(topic), "Script", review)
    os.makedirs(f"{storage}/content/")
    util.write_file(content, f"{storage}/content/content.txt")
    return content


def create_prompts(storage: str, content: str, n: int, review: bool) -> List[str]:
    prompts = reviewed(
        lambda: TextModel().generate_image_prompts(content, n), "Image prompts", review
    )
    util.write_file(str(prompts), f"{storage}/content/prompts.json")
    return prompts


def create_images(storage: str, prompts: List[str]) -> str:
    image_storage = f"{storage}/images/"
    os.makedirs(image_storage)

    image_model = ImageModel(image_storage)
    timestamp = str(time.time())

    image_model.generate_images(prompts, timestamp)

    return image_storage


def create_audio(storage: str, content: str) -> str:
    audio_storage = f"{storage}/audio/"
    os.makedirs(audio_storage)

    audio_model = AudioModel(audio_storage)
    timestamp = str(time.time())
    file_path = audio_model.generate_audio(content, timestamp)

    return file_path


def create_subtitle(audio_filepath: str) -> List:
    subtitle_model = SubtitleModel()
    return subtitle_model.generate_subtitle(audio_filepath)


def create_video(
    storage: str,
    audio_path: str,
    image_folder: str,
    subtitle: Optional[List],
    bgm_path: Optional[str],
):
    video_path = f"{storage}/video.mp4"
    video_model = VideoModel(audio_path)
    video_model.set_video(image_folder)
    video_model.attach_audio(bgm_path)
    if subtitle:
        video_model.attach_subtitle(subtitle)
    video_model.generate_video(video_path)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # emoji logs crash cp1252 when piped on Windows
    parser = argparse.ArgumentParser(description="Generate a short video offline.")
    parser.add_argument("topic", nargs="?", default=random.choice(DEFAULT_TOPICS))
    parser.add_argument(
        "--review", action="store_true", help="approve script and prompts before rendering"
    )
    args = parser.parse_args()

    storage = f"outputs/{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"🎬 Topic: {args.topic} -> {storage}")

    content = create_script(storage, args.topic, args.review)

    audio_path = create_audio(storage, content)
    util.free_gpu()

    subtitle = create_subtitle(audio_path)
    util.free_gpu()

    n_images = max(3, round(soundfile.info(audio_path).duration / SECONDS_PER_IMAGE))
    image_prompts = create_prompts(storage, content, n_images, args.review)

    images_folder = create_images(storage, image_prompts)
    util.free_gpu()

    create_video(
        storage=storage,
        audio_path=audio_path,
        image_folder=images_folder,
        subtitle=subtitle,
        bgm_path="bgms/classic.mp3",
    )


if __name__ == "__main__":
    main()
