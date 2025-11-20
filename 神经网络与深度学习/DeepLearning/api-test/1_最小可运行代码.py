from google import genai

client = genai.Client(api_key="AIzaSyC3RWmcGvRDfLG0QtC4QUHq0ZhAQFpO7Vc")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="用一句话解释什么是深度学习。",
)

print(response)
print(response.text)
