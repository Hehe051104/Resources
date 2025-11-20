from google import genai

client = genai.Client(api_key="AIzaSyC3RWmcGvRDfLG0QtC4QUHq0ZhAQFpO7Vc")

stream = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="解释深度学习，但是流式输出。",
)

for chunk in stream:
    if chunk.text:
        print(chunk.text, end="", flush=True)