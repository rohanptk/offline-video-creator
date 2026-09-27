import os
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *

FRAME_SIZE = (480, 848)
ZOOM = 0.08  # Ken Burns: grow 8% over each image
HIGHLIGHT = "#FFD700"
FONT_SIZE = 50


def load_font(size: int):
    for name in ("impact.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def group_subtitles(subtitles: List, group_size: int = 4) -> List:
    # whisperx omits start/end for words it can't align (numbers, symbols):
    # keep their text, time the group from its aligned words.
    grouped = []

    for i in range(0, len(subtitles), group_size):
        group = subtitles[i : i + group_size]
        timed = [w for w in group if "start" in w and "end" in w]
        if not timed:
            continue
        grouped.append(
            {
                "word": " ".join([w["word"] for w in group]),
                "start": timed[0]["start"],
                "end": timed[-1]["end"],
                "words": group,
            }
        )

    return grouped


def caption_segments(words: List, end: float) -> List:
    """(highlight_idx, start, end) per timed word; each runs until the next
    timed word starts, the last until the group ends. Untimed words never light up."""
    timed = [i for i, w in enumerate(words) if "start" in w and "end" in w]
    return [
        (i, words[i]["start"], words[timed[k + 1]]["start"] if k + 1 < len(timed) else end)
        for k, i in enumerate(timed)
    ]


def render_caption(words: List[str], highlight_idx: Optional[int], max_width: int, font=None):
    """RGBA caption: wrapped words on a translucent box, one word highlighted."""
    font = font or load_font(FONT_SIZE)
    pad, stroke = 12, 2
    space = font.getlength(" ")
    line_h = font.size + 8

    lines, line, line_w = [], [], 0.0  # greedy wrap, keeping word indices
    for i, word in enumerate(words):
        w = font.getlength(word)
        if line and line_w + space + w > max_width - 2 * pad:
            lines.append((line, line_w))
            line, line_w = [], 0.0
        line_w += (space if line else 0) + w
        line.append(i)
    lines.append((line, line_w))

    width = int(max(lw for _, lw in lines)) + 2 * pad
    img = Image.new("RGBA", (width, line_h * len(lines) + 2 * pad), (0, 0, 0, 102))
    draw = ImageDraw.Draw(img)
    for row, (idxs, lw) in enumerate(lines):
        x = (width - lw) / 2
        for i in idxs:
            draw.text(
                (x, pad + row * line_h),
                words[i],
                font=font,
                fill=HIGHLIGHT if i == highlight_idx else "white",
                stroke_width=stroke,
                stroke_fill="black",
            )
            x += font.getlength(words[i]) + space
    return np.array(img)


class VideoModel:
    def __init__(self, audio_path: str):
        self.audio = AudioFileClip(audio_path)
        self.audio_duration = self.audio.duration

    def set_video(self, image_folder: str):
        image_files = sorted(
            [
                os.path.join(image_folder, fname)
                for fname in os.listdir(image_folder)
                if fname.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        image_duration = self.audio_duration / len(image_files)
        image_clips = [
            VideoEffects.ken_burns(path, image_duration) for path in image_files
        ]
        base_video = concatenate_videoclips(image_clips, method="compose", padding=-0.5)

        # Calculate leftover time
        leftover = self.audio_duration - base_video.duration
        if leftover <= 0:
            self.video = base_video
            return

        filler_clip = (
            ImageClip("images/filler.png")
            .set_duration(leftover)
            .resize(base_video.size)
            .fadein(0.2)
            .fadeout(0.2)
        )

        self.video = concatenate_videoclips([base_video, filler_clip], method="compose")

    def attach_audio(self, bgm_path: str | None = None):
        audios = [self.audio]
        if bgm_path:
            bgm = AudioFileClip(bgm_path).volumex(0.2).set_duration(self.audio_duration)
            audios.append(bgm)
        composite_audio = CompositeAudioClip(audios)
        self.video = self.video.set_audio(composite_audio)

    def attach_subtitle(self, subtitles: List):
        self.video = SubtitleEffects.simple_subtitle(
            self.video, group_subtitles(subtitles)
        )

    def generate_video(self, output_path: str):
        self.video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            audio=True,
        )

        print(f"🔊 Has audio: {self.video.audio is not None}")


class VideoEffects:
    @staticmethod
    def ken_burns(image_path, duration, fade_duration=0.5):
        clip = ImageClip(image_path)
        w, h = FRAME_SIZE
        cover = max(w / clip.w, h / clip.h)  # fill the frame, crop the overflow
        zoomed = clip.set_duration(duration).resize(
            lambda t: cover * (1 + ZOOM * t / duration)
        )
        return (
            CompositeVideoClip([zoomed.set_position("center")], size=FRAME_SIZE)
            .set_duration(duration)
            .fadein(fade_duration)
            .fadeout(fade_duration)
        )


class SubtitleEffects:
    @staticmethod
    def simple_subtitle(video, subtitle: List):
        font = load_font(FONT_SIZE)
        subtitle_clips = []
        for group in subtitle:
            texts = [w["word"] for w in group["words"]]
            for idx, start, end in caption_segments(group["words"], group["end"]):
                caption = render_caption(texts, idx, video.w - 100, font)
                subtitle_clips.append(
                    ImageClip(caption)
                    .set_position(("center", "center"))
                    .set_start(start)
                    .set_duration(end - start)
                )

        return CompositeVideoClip([video] + subtitle_clips)
