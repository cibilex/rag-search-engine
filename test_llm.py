import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("DEEPSEEK_TOKEN")
if not api_key:
    raise RuntimeError("DEEPSEEK_TOKEN environment variable not set")

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "user",
            "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
        }
    ],
)

print(response.choices[0].message.content)
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Response tokens: {response.usage.completion_tokens}")
