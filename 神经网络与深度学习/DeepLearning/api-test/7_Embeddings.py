from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyC3RWmcGvRDfLG0QtC4QUHq0ZhAQFpO7Vc")

r = client.models.embed_content(
    model="gemini-embedding-001",
    contents=types.Content(
        role="user",
        parts=[types.Part(text="这是要嵌入的文本。")]
    )
)

# 嵌入向量在 r.embeddings[0].values
print(r.embeddings[0].values[:10])
