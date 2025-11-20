from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyC3RWmcGvRDfLG0QtC4QUHq0ZhAQFpO7Vc")

with open("document/test.pdf", "rb") as f:
    pdf_bytes = f.read()

res = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=[
        types.Part.from_bytes(
            data=pdf_bytes,
            mime_type="application/pdf"
        ),
        types.Part(text="总结这个 PDF 的核心内容,我需要做什么防止挂科!")
    ]
)

print(res.text)
