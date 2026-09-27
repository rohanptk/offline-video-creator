import json
import os
import ollama
from typing import List

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


class TextModel:
    def generate_content(self, topic: str="an historical inspiring fact that changed today's computer science world") -> str:

        system_message = """
        You are a YouTube Shorts Assistant help me to generate content. 
        No explanation, No additional text, No additional decorations, No screenplay, only content as plain paragraph but include commas, exclamations, question marks accordingly. 
        Use simple english, No roman representations.
        Keep it family-friendly: no nudity, sexual content or gore.
        Ask for like, share and subscribe at the end of content.
        """

        user_message = f"""
        Create a 40-second content about {topic},

        Did you know...
        """

        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()},
            ],
            keep_alive=0,  # unload from VRAM right away, SD needs it next
        )

        content = response["message"]["content"]
        final_response = f"Welcome to the channel, {content}"

        return final_response

    def generate_image_prompts(self, content: str, n: int = 5) -> List[str]:
        system_message = """
        You are an artist and a prompt engineer who could describe image in words clearly by a content sequentially. 
        No explanation, No additional text, No additional decorations.
        Keep every image family-friendly: fully clothed people, no nudity, sexual content, gore or blood.

        Respond strictly as JSON: {"prompts": ["...", "..."]}
        """

        user_message = f"""
        Give list of {n} descriptive, clear image prompts for below content sequentially

        {content}
        """

        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()},
            ],
            format="json",
            keep_alive=0,
        )

        content = response["message"]["content"]
        return list(json.loads(content)["prompts"])
