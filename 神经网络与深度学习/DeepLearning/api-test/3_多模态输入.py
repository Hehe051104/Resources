from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyC3RWmcGvRDfLG0QtC4QUHq0ZhAQFpO7Vc")

with open("img/1.jpg", "rb") as f:
    img_bytes = f.read()

res = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        "图中的人是谁?",
        types.Part.from_bytes(data=img_bytes,mime_type='image/jpeg')   #要指明输入的图片格式  mime:多用途互联网邮件扩展类型
    ]
)

print(res.text)