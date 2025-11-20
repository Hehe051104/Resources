from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyC3RWmcGvRDfLG0QtC4QUHq0ZhAQFpO7Vc")

with open("audio/test.wav", "rb") as f:
    audio_bytes = f.read()

res= client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "这首歌叫什么?",
        types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav"
        )
    ]
)

print(res.text)
