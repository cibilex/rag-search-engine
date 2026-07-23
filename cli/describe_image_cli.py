import argparse
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_PROMPT = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_TOKEN")
    if not api_key:
        raise RuntimeError("OPENROUTER_TOKEN environment variable not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Query Rewriting CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file")
    parser.add_argument("--query", type=str, required=True, help="Text query to rewrite")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()

    client = get_client()

    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
