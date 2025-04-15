# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-3B-Instruct")
pipe(messages)