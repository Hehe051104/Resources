from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyC3RWmcGvRDfLG0QtC4QUHq0ZhAQFpO7Vc")

chat = client.chats.create(
    model="gemini-2.5-flash",
    history=[
        types.Content(
            role="user",
            parts=[types.Part(text="你好")]
        ),
        types.Content(
            role="model",
            parts=[types.Part(text="你好呀")]
        )
    ]
)

reply = chat.send_message("你能做什么？")
print(reply.text)
